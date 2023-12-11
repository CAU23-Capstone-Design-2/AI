# Usage: python pre.py --url https://www.youtube.com/watch?v=6Dh-RL__uN4 --music_name IU_gooday

import argparse
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str , help='youtube url', required=True)
#    parser.add_argument('--music_name', type=str, help='music name',required=True)
    # music_code : youtube url 에서 추출
    # https://www.youtube.com/watch?v=w0cAt8zZJ10&list=abc&num=3 -> w0cAt8zZJ10
    args = parser.parse_args()
    # v 파라미터값 저장
    music_code = args.url.split('=')[1].split('&')[0]
    music_code = music_code.replace('"','')
    music_code = music_code.strip()
#    music_code    
    
    # 1. mypytube 로 음원 다운로드
    try:
        print('============== Start mypytube ============== ')
        command = ['python3', 'mypytube.py', '--url', args.url, '--music_name', music_code]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print('============== End mypytube ============== ')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.stdout)
        print(e.stderr)
        raise e
    
    # 2-1. uvr 로 음원 분리 ( vocal & instrumental seperation )
    # model : HP2_all_vocals
    try:
        print('============== Start uvr HP2_all_vocals ============== ')
        command = ['python3', 'myuvr.py', '--model_name', 'HP2_all_vocals', '--inp_path', f'./music/{music_code}.wav', '--save_root_vocal', './vocal', '--save_root_inst', './inst', '--agg', '10', '--format0', 'wav']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        # 파일명 변경
        # ./vocal/vocal_{music_code}.wav_10.wav -> ./vocal/{music_code}.wav
        os.rename(f'./vocal/vocal_{music_code}.wav_10.wav', f'./vocal/{music_code}.wav')
        os.rename(f'./inst/instrument_{music_code}.wav_10.wav', f'./inst/{music_code}.wav')
        print(f"output: ./vocal/{music_code}.wav")
        print(f'output : ./inst/{music_code}.wav')
        print('============== End uvr HP2_all_vocals ============== ')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.stdout)
        print(e.stderr)
        raise e
    
    
    '''
    # 2-2 uvr로 음원 분리 ( dereverb )
    try:
        print('============== Start uvr VR-DeEchoAggressive.pth ============== ')
        command = ['python3', 'myuvr.py', '--model_name', 'VR-DeEchoAggressive.pth', '--inp_path', f'./vocal/{music_code}.wav', '--save_root_vocal', './inst', '--save_root_inst', './vocal', '--agg', '10', '--format0', 'wav']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        os.rename(f'./vocal/{music_code}.wav_main_vocal.wav', f'./dereverb/{music_code}.wav')
        os.remove(f'./inst/{music_code}.wav_others.wav')
        print(f"output: ./dereverb/{music_code}.wav")
        print('============== End uvr VR-DeEchoAggressive.pth ============== ')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.stdout)
        print(e.stderr)
        raise e
    '''
    
    # 2-2 uvr로 음원 분리 ( dereverb )
    try:
        print('============== Start uvr VR-DeEchoAggressive.pth ============== ')
        command = ['python3', 'myuvr.py', '--model_name', 'VR-DeEchoAggressive', '--inp_path', f'./vocal/{music_code}.wav', '--save_root_vocal', './inst', '--save_root_inst', './vocal', '--agg', '10', '--format0', 'wav']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        os.rename(f'./inst/instrument_{music_code}.wav_10.wav', f'./dereverb/{music_code}.wav')
        os.remove(f'./vocal/vocal_{music_code}.wav_10.wav')
        print(f"output: ./dereverb/{music_code}.wav")
        print('============== End uvr VR-DeEchoAggressive.pth ============== ')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.stdout)
        print(e.stderr)
        raise e