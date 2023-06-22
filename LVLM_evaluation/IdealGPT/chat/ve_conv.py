from .vcr_conv import VCRConversationTwoAgent
from .call_gpt import call_gpt
from .ve_prompt import *
import re
from tqdm import tqdm
import copy
import time

class VEConversationTwoAgent(VCRConversationTwoAgent):
    def __init__(self, img, vqa_model, model, question, answer_choices, data_id, prompt_setting='v1a', caption=None, temp_gpt=0):
        super().__init__(img, vqa_model, model, question, answer_choices, prompt_setting, caption, temp_gpt)
        if type(self) == VEConversationTwoAgent:
            assert prompt_setting in ['v1a']

        self.data_id = data_id
        self.prompt_setting = prompt_setting
        self.chat_history = {}
        self.chat_history_init_asker = []
        self.chat_history_more_asker = []
        self.chat_history_reasoner = []
        # add prompts.
        if type(self) == VEConversationTwoAgent:
            if prompt_setting == 'v1a':
                self.INIT_ASKER_SYSTEM_PROMPT = INIT_ASKER_SYSTEM_PROMPT_V1
                self.INIT_ASKER_FIRST_QUESTION = INIT_ASKER_FIRST_QUESTION_V1

                self.MORE_ASKER_SYSTEM_PROMPT = MORE_ASKER_SYSTEM_PROMPT_V1
                self.MORE_ASKER_FIRST_QUESTION = MORE_ASKER_FIRST_QUESTION_V1

                self.REASONER_SYSTEM_PROMPT = REASONER_SYSTEM_PROMPT_V1A
                self.REASONER_FIRST_QUESTION = REASONER_FIRST_QUESTION_V1A
                self.FINAL_REASONER_SYSTEM_PROMPT = FINAL_REASONER_SYSTEM_PROMPT_V1A
            else:
                raise NotImplementedError(f'{prompt_setting} not supported in class VEConversationTwoAgent.')

    def prepare_init_asker_message(self, prompt, caption, question, answer_choices): # ['entailment', 'neutral', 'contradiction']
        answer_prompt = ''
        for ans_id, ans_str in enumerate(answer_choices):
            answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

        if self.prompt_setting in ['v1a']:
            input_prompt = 'Imperfect Caption: {}\nHypothesis: {}\nThree choices:\n{}'.format(caption, question, answer_prompt)
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
            input_prompt = 'Imperfect Caption: {}\nHypothesis: {}\nThree choices: \n{} Sub-questions and answers: \n{} Analysis: \n{}'.format(
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
            input_prompt = 'Imperfect Caption: {}\nHypothesis: {}\nThree choices: \n{} Existing Sub-questions and answers: \n{}'.format(
                caption, question, answer_prompt, sub_answer_prompt)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages
    

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
        matches = re.findall(pattern, gpt_response) # [' contradiction.']
        if matches:
            answer_string = matches[-1].strip() # contradiction.
            # In middle rounds, detect 'not sure' at first.
            if not final_round_flag:
                if 'not sure' in answer_string.lower():
                    answer_string = None

            # If 'not sure' is not detected, detect number.
            if answer_string is not None:
                answer_string = re.sub(r'[^\w\s]', '', answer_string) # contradiction
                # If a number is found, return it; otherwise, return None
                if answer_string in self.answer_choices:
                    answer_string = answer_string  
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
        if gpt_answer in self.answer_choices:
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
            max_try = 10
            # Parse predicted answer from GPT output if any.
            if round_i == max_n_rounds - 1:
                final_round_flag = True
            else:
                final_round_flag = False
            while try_num < max_try:
                try_num += 1
                success = False
                while not success:                
                    try:
                        gpt_response, n_tokens = call_gpt(self.chat_history_reasoner, model=self.model, temp_gpt=self.temp_gpt)
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