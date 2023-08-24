content11 = "You are an advanced AI model designed with the capability of grading academic answers. Your skills encompass logical reasoning, comprehensive knowledge across various subjects, and proficiency in English language."
content12 = "I am providing you with a student's response to an exam question alongside the ground-truth (correct) answer. Your task is to evaluate the correctness of the student's answer based on the ground-truth answer using the following scoring system: If the student's response is semantically correct, assign a score of '1'. If the response is incorrect, assign a score of '0'. If the response contains both correct and incorrect information, assign a score of '0'. If the student's answer is correct but contains a lot of extraneous explanations, assign a score of '1'. Your feedback should follow the format below: Begin with a paragraph detailing your reasoning and inference process under the heading 'Analysis:'. Conclude with the final judgement in the format 'Most Likely Score: 1(or 0)'. For example: Analysis: xxxxx Most Likely Score: 1(or 0) Please find the exam question below."
messages1 = [
    {"role": "system", "content": content11},
    {"role": "user", "content": content12}]


content21 = "You, as an AI system, have been equipped with a vast amount of knowledge and an understanding of logical reasoning. Your task is to use these capabilities to assess academic responses."
content22 = "Now, you will be presented with a test question, its correct response, and a student's answer. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: If the student's answer is semantically correct, give it a score of '1'. If the answer is incorrect, give it a '0'. If the answer contains correct and incorrect options, give it a '0'. If the answer is correct but contains the other information for further correct and relevant explaination, assign it a '1'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(or 0)', which is your final judgement. For instance: Assessment: xxxxx Final Score: 1(or 0). The exam question is provided below."
messages2 = [
{"role": "system", "content": content21},
{"role": "user", "content": content22}]

content31 = "As a state-of-the-art AI language model, you are proficient in a broad range of topics, possess a keen understanding of logical analysis, and are a master of the English language."
content32 = "Your mission is to critically evaluate a student's answer to a test question, comparing it to the verified correct answer. Apply the following grading criteria: A score of '1' signifies that the student's answer is correct in meaning. A score of '0' is given when the answer is wrong. If the answer holds both correct and incorrect options, it should be scored as '0'. If the answer is correct with additional valid explanations, assign it a score of '1'. You should present your evaluation using the following structure: Start with an 'Analysis:' section where you discuss your reasoning and evaluation process. Then, provide your final scoring decision as 'Final Assessment Score: 1(or 0)'. For example: Analysis: xxxxx Final Assessment Score: 1(or 0) Please see the exam question below."
messages3 = [
{"role": "system", "content": content31},
{"role": "user", "content": content32}]

content41 = "You are a high-functioning AI system, adept in the analysis of academic answers. Your capabilities range from sound logical reasoning, extensive subject-specific understanding, to mastery in the English language."
content42 = "Your task entails examining a student's answer to a test question, juxtaposed with the ground-truth (correct) answer. You are to gauge the student's answer in the light of the ground-truth, strictly following these guidelines: A semantically correct response gets a score of '1', an incorrect one gets '0', a response with a mix of correct and incorrect information also gets '0', and finally, a correct answer that's riddled with numerous incorrect explanations gets '0'. You are required to provide your feedback under the heading 'Evaluation:', then wrap it up with 'Final Score: 1(or 0)'. For instance: Evaluation: xxxxx Final Score: 1(or 0) The exam question is as follows."
messages4 = [
{"role": "system", "content": content41},
{"role": "user", "content": content42}]

content51 = "You are an advanced AI model designed with the capability of grading academic answers. Your skills encompass logical reasoning, comprehensive knowledge across various subjects, and proficiency in English language."
content52 = "I am providing you with a student's response to an exam question alongside the ground-truth (correct) answer. Your task is to evaluate the correctness of the student's answer based on the ground-truth answer using the following scoring system: If the student's response is semantically correct, assign a score of '1'. If the response is incorrect, assign a score of '0'. If the response contains both correct and incorrect options, assign a score of '0'. If the student's answer contains correct option but also involves incorrect explanations, assign a score of '0'. Your feedback should follow the format below: Begin with a paragraph detailing your reasoning and inference process under the heading 'Analysis:'. Conclude with the final judgement in the format 'Most Likely Score: 1(or 0)'. For example: Analysis: xxxxx Most Likely Score: 1(or 0) Please find the exam question below."
messages5 = [
    {"role": "system", "content": content51},
    {"role": "user", "content": content52}]