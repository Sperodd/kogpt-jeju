import os
import json
import re
from glob import glob


class SentenceDataLoader:
    @staticmethod
    def give_data(file):
        with open(file, "r", encoding="utf-8") as f:
                tmp = json.load(f)
        data = []
        for row in tmp.get('utterance'):
            if row.get('form') == row.get('standard_form'):
                continue
            
            eojeols = []
            for eojeol in row.get('eojeolList'):
                word = eojeol.get('eojeol')

                if word.startswith('{'): 
                    continue

                word = word.split('/')[0].replace('(', '').replace(')', '').replace('#', '')
                word = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣]', '', word).strip()
                if not word:
                    continue
                eojeols.append(word)
            data.append(" ".join(eojeols))
        return data

    def make_txt(self, max_file_num=300):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "data", "한국어_방언_발화_데이터(제주도)", "(비식별화완료)제주도_학습용데이터_1")
    
        files = glob(os.path.join(data_path, "*.json"))
        print(f"총 파일 수: {len(files)}")
    
        all_data = []
        for file in files[:max_file_num]:
            for row in self.give_data(file):
                all_data.append(row)
    
        print(f'총 문장 수: {len(all_data)}')
        with open(os.path.join(current_dir, 'sentences.txt'), 'w', encoding='utf-8') as file:
            file.writelines(line + "\n" for line in all_data if line)
        print(f'파일 생성 완료')

if __name__ == "__main__":
    dataLoader = SentenceDataLoader()
    dataLoader.make_txt()