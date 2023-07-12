from tqdm import tqdm
import re
from copy import deepcopy
from .call_gpt import call_gpt
from .vcr_prompt import *
import time

class VCRConversationTwoAgent():
    def __init__(self, img, vqa_model, model, question, answer_choices, data_id, prompt_setting='v1a', caption=None, temp_gpt=0):
        if type(self) == VCRConversationTwoAgent:
            assert prompt_setting in ['v1a']

        self.img = img
        self.vqa_model = vqa_model
        self.model = model

        self.question = question
        self.answer_choices = answer_choices
        self.answer_predict = None

        self.catpion = caption
        self.sub_questions = []
        self.sub_answers = []
        self.chat_history = []

        self.total_tokens = 0
        self.temp_gpt=temp_gpt

        self.prompt_setting = prompt_setting
        self.use_caption = True


        self.data_id = data_id
        self.chat_history = {}
        self.chat_history_init_asker = []
        self.chat_history_more_asker = []
        self.chat_history_reasoner = []
        # add prompts.
        if type(self) == VCRConversationTwoAgent:
            if prompt_setting == 'v1a':
                self.INIT_ASKER_SYSTEM_PROMPT = INIT_ASKER_SYSTEM_PROMPT_V1
                self.INIT_ASKER_FIRST_QUESTION = INIT_ASKER_FIRST_QUESTION_V1

                self.MORE_ASKER_SYSTEM_PROMPT = MORE_ASKER_SYSTEM_PROMPT_V1
                self.MORE_ASKER_FIRST_QUESTION = MORE_ASKER_FIRST_QUESTION_V1

                self.REASONER_SYSTEM_PROMPT = REASONER_SYSTEM_PROMPT_V1A
                self.REASONER_FIRST_QUESTION = REASONER_FIRST_QUESTION_V1A
                self.FINAL_REASONER_SYSTEM_PROMPT = FINAL_REASONER_SYSTEM_PROMPT_V1A
            else:
                raise NotImplementedError(f'{prompt_setting} not supported in class VCRConversationTwoAgent.')

        blip2_QA_prompt = 'Question: placeholder Answer:'
        llava_QA_prompt = 'placeholder Reply in short.'
        minigpt4_QA_prompt = 'placeholder Answer it in one short sentence.'

        if 'llava' in self.vqa_model.model_type:
            self.vqa_prompt =  llava_QA_prompt
        elif 'minigpt4' in self.vqa_model.model_type:
            self.vqa_prompt = minigpt4_QA_prompt
        elif 't5' in self.vqa_model.model_type or 'opt' in self.vqa_model.model_type:
            self.vqa_prompt =  blip2_QA_prompt         
        else:
            raise NotImplementedError(f'Could not find vqa prompt for {self.vqa_model.model_type}.')
        

    def prepare_init_asker_message(self, prompt, caption, question, answer_choices):
        answer_prompt = ''
        for ans_id, ans_str in enumerate(answer_choices):
            answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

        if self.prompt_setting in ['v1a']:
            input_prompt = 'Imperfect Caption: {}\nMain Question: {}\nFour choices:\n{}'.format(caption, question, answer_prompt)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages


    def prepare_more_asker_message(self, prompt, caption, question, answer_choices, sub_questions, sub_answers, analysis):
        answer_prompt = ''
        for ans_id, ans_str in enumerate(answer_choices):
            answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

        sub_answer_prompt = ''
        flat_sub_questions = []
        for sub_questions_i in sub_questions:
            flat_sub_questions.extend(sub_questions_i)
        flat_sub_answers = []
        for sub_answers_i in sub_answers:
            flat_sub_answers.extend(sub_answers_i)

        assert len(flat_sub_questions) == len(flat_sub_answers)
        for ans_id, ans_str in enumerate(flat_sub_answers):
            sub_answer_prompt = sub_answer_prompt + 'Sub-question: {} Answer: {}\n'.format(flat_sub_questions[ans_id], ans_str)
            
        if self.prompt_setting in ['v1a']:
            input_prompt = 'Imperfect Caption: {}\nMain Question: {}\nFour choices: \n{} Sub-questions and answers: \n{} Analysis: \n{}'.format(
                caption, question, answer_prompt, sub_answer_prompt, analysis)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages


    def prepare_reasoner_message(self, prompt, caption, question, answer_choices, sub_questions, sub_answers):
        answer_prompt = ''
        for ans_id, ans_str in enumerate(answer_choices):
            answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

        sub_answer_prompt = ''
        flat_sub_questions = []
        for sub_questions_i in sub_questions:
            flat_sub_questions.extend(sub_questions_i)
        flat_sub_answers = []
        for sub_answers_i in sub_answers:
            flat_sub_answers.extend(sub_answers_i)

        assert len(flat_sub_questions) == len(flat_sub_answers)
        for ans_id, ans_str in enumerate(flat_sub_answers):
            sub_answer_prompt = sub_answer_prompt + 'Sub-question: {} Answer: {}\n'.format(flat_sub_questions[ans_id], ans_str)
            
        if self.prompt_setting in ['v1a']:
            input_prompt = 'Imperfect Caption: {}\nMain Question: {}\nFour choices: \n{} Existing Sub-questions and answers: \n{}'.format(
                caption, question, answer_prompt, sub_answer_prompt)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages
    

    def answer_question(self, cur_sub_questions):
        # prepare the context for blip2
        sub_answers = []
        for sub_question_i in cur_sub_questions:
            vqa_prompt = self.vqa_prompt.replace('placeholder', sub_question_i)
            # Feed into VQA model.
            if 'llava' in self.vqa_model.model_type or 'minigpt4' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt)
            elif 't5' in self.vqa_model.model_type or 'opt' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt, length_penalty=-1, max_length=10)
            else:
                raise NotImplementedError(f'Not support VQA of {self.vqa_model.model_type}.')

            answer = self.answer_trim(answer)
            sub_answers.append(answer)
        return sub_answers


    def answer_trim(self, answer):
        answer = answer.split('Question:')[0].replace('\n', ' ').strip()
        return answer

    def parse_subquestion(self, gpt_response):
        gpt_response = gpt_response + '\n'
        sub_questions = []
        while True:
            result = re.search('Sub-question.{0,3}:(.*)\n', gpt_response)
            if result is None:
                break
            else:
                sub_questions.append(result.group(1).strip())
                gpt_response = gpt_response.split(result.group(1))[1]

        return sub_questions

    def parse_final_answer(self, gpt_response, final_round_flag):
        # ===== Parse the paragraph starting with analysis. =====
        analysis_result = re.search('Analysis:(.*)\n', gpt_response)
        if analysis_result:
            analysis_string = analysis_result.group(1).strip()
        else:
            print(f'Can not parse analysis from {gpt_response}')
            raise ValueError

        if self.prompt_setting in ['v1a']:
            substring = "More Likely Answer:"

        pattern = f"{re.escape(substring)}(.*?)(?=\n|$)"
        matches = re.findall(pattern, gpt_response)
        if matches:
            answer_string = matches[-1].strip()
            # In middle rounds, detect 'not sure' at first.
            if not final_round_flag:
                if 'not sure' in answer_string.lower():
                    answer_string = None

            # If 'not sure' is not detected, detect number.
            if answer_string is not None:
                # Search for the first occurrence of a number (continuous digits) in the string
                number_match = re.search(r'\d+', answer_string)
                # If a number is found, return it; otherwise, return None
                if number_match:
                    answer_string = number_match.group(0)  
                else:
                    answer_string = None
        else:
            print(f'Can not parse Predicted Answer: from {gpt_response}')
            raise ValueError

        return analysis_string, answer_string
    
    
    def parse_final_answer_rerun(self, gpt_response, final_round_flag):
        try:
            analysis_string, answer_string = self.parse_final_answer(gpt_response=gpt_response, final_round_flag=final_round_flag)
            need_rerun = False
            return analysis_string, answer_string, need_rerun
        except:
            need_rerun = True
            return None, None, need_rerun


    def break_condition(self, gpt_answer):
        if gpt_answer in ANSWER_MAP:
            return True
        else:
            return False


    def chatting(self, max_n_rounds, print_mode):
        # Caption first.
        if self.catpion is None and self.use_caption:
            self.catpion = self.vqa_model.caption(self.img)
        
        self.chat_history = {'init_asker':[],
                             'more_asker':[],
                             'reasoner':[]}
        for round_i in tqdm(range(max_n_rounds), desc='Chat Rounds', disable=print_mode != 'bar'):
            if round_i == 0:
                # Prepare initial gpt input for decomposing into sub-questions, and Update chat_history.
                assert self.catpion != None if self.use_caption else self.catpion == None

                self.chat_history_init_asker = [{"role": "system", "content": self.INIT_ASKER_SYSTEM_PROMPT}]
                gpt_input = self.prepare_init_asker_message(prompt=self.INIT_ASKER_FIRST_QUESTION, caption=self.catpion, question=self.question, answer_choices=self.answer_choices)
                self.chat_history_init_asker.append(gpt_input)

                # Run GPT and update chat_history.
                success = False
                while not success:
                    try:
                        gpt_response, n_tokens = call_gpt(self.chat_history_init_asker, model=self.model, temp_gpt=self.temp_gpt)
                        success = True
                    except Exception as e:
                        print('[Worker] an exception occured: %s (%s). retrying in 3 seconds.' % (type(e), str(e)))
                        time.sleep(3)                        
                self.chat_history_init_asker.append({'role': 'assistant', 'content': gpt_response})
                self.total_tokens = self.total_tokens + n_tokens

                # Save history
                self.chat_history['init_asker'].append(self.chat_history_init_asker)

            else:
                # GPT is not sure, let GPT ask additional questions, and update chat_history.
                self.chat_history_more_asker = [{"role": "system", "content": self.MORE_ASKER_SYSTEM_PROMPT}]
                gpt_input = self.prepare_more_asker_message(prompt=self.MORE_ASKER_FIRST_QUESTION, caption=self.catpion, question=self.question, answer_choices=self.answer_choices, 
                                                            sub_questions=self.sub_questions, sub_answers=self.sub_answers, analysis=cur_analysis)
                self.chat_history_more_asker.append(gpt_input)

                # Run GPT.
                success = False
                while not success:
                    try:
                        gpt_response, n_tokens = call_gpt(self.chat_history_more_asker, model=self.model, temp_gpt=self.temp_gpt)
                        success = True
                    except Exception as e:
                        print('[Worker] an exception occured: %s (%s). retrying in 3 seconds.' % (type(e), str(e)))
                        time.sleep(3)                      
       
                self.chat_history_more_asker.append({'role': 'assistant', 'content': gpt_response})
                self.total_tokens = self.total_tokens + n_tokens

                # Save history
                self.chat_history['more_asker'].append(self.chat_history_more_asker)


            #  Post process GPT response to get sub-questions.
            cur_sub_questions = self.parse_subquestion(gpt_response)
            # if len(cur_sub_questions) != 0:
            self.sub_questions.append(cur_sub_questions)

            # Use VQA model to answer sub-questions.
            cur_sub_answers = self.answer_question(cur_sub_questions)
            self.sub_answers.append(cur_sub_answers) 

            # Input sub-questions and sub-answers into a reasoner GPT.
            if round_i == max_n_rounds - 1:
                self.chat_history_reasoner = [{"role": "system", "content": self.FINAL_REASONER_SYSTEM_PROMPT}]
            else:
                self.chat_history_reasoner = [{"role": "system", "content": self.REASONER_SYSTEM_PROMPT}]
            gpt_input = self.prepare_reasoner_message(prompt=self.REASONER_FIRST_QUESTION, caption=self.catpion, question=self.question, answer_choices=self.answer_choices,
                                                      sub_questions=self.sub_questions, sub_answers=self.sub_answers)
            self.chat_history_reasoner.append(gpt_input)

            # Run GPT.
            try_num = 0
            max_try = 15
            # Parse predicted answer from GPT output if any.
            if round_i == max_n_rounds - 1:
                final_round_flag = True
            else:
                final_round_flag = False
            while try_num < max_try:
                try_num += 1
                if try_num > max_try/2:
                    cur_temp_gpt = self.temp_gpt + 0.1 * ((2.0*try_num/max_try)-1)
                else:
                    cur_temp_gpt = self.temp_gpt
                success = False
                while not success:
                    try:
                        gpt_response, n_tokens = call_gpt(self.chat_history_reasoner, model=self.model, temp_gpt=cur_temp_gpt)
                        success = True
                    except Exception as e:     
                        print('[Worker] an exception occured: %s (%s). retrying in 3 seconds.' % (type(e), str(e)))
                        time.sleep(3)                      
              
                self.total_tokens = self.total_tokens + n_tokens

                cur_analysis, gpt_answer, need_rerun = self.parse_final_answer_rerun(gpt_response, final_round_flag=final_round_flag)
                if not need_rerun:
                    break
                else:
                    if try_num == max_try:
                        raise ValueError('Rerun too many times, still failed in parsing.')
                    else:
                        print(f'Parsing failed, Time {try_num} of Rerun GPT Decision for data {self.data_id}.')

            # Save history
            self.chat_history_reasoner.append({'role': 'assistant', 'content': gpt_response})
            self.chat_history['reasoner'].append(self.chat_history_reasoner)

            self.answer_predict = gpt_answer

            # If gpt answer satisfies some condition. Finish current loop.
            if self.break_condition(gpt_answer=gpt_answer):
                break
                  
        return round_i+1