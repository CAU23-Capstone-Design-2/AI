# Usage: python mypytube.py --url https://www.youtube.com/watch?v=6Dh-RL__uN4 --music_name IU_gooday

import os
import argparse
import pytube


ROOT_DIR = os.path.abspath('./')

def pytube(url,name):
    from pytube import YouTube
    from pydub import AudioSegment

    try:
        yt = YouTube(url)
    except:
        print("Connection Error")

    # 영상 다운로드, music/{name} 저장
    yt.streams.filter(only_audio=True).first().download(filename = f"./music/{name}")

    # 다운로드된 음서파일을 wav 파일로 변환
    AudioSegment.from_file(f'./music/{name}').export(f'./music/{name}.wav', format='wav')
    os.remove(f"./music/{name}")

    print('Download {} Complete'.format(yt.title))
    print('Saved at ./music/{}.wav'.format(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str , help='youtube url', required=True)
    parser.add_argument('--music_name', type=str, help='music name', required=True)
    args = parser.parse_args()

    # 1. download
    print('1. download')
    pytube(url = args.url, name = args.music_name)

    # 2. vocal remover
