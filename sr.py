import wave

def get_wave_file_sample_rate(file_path):
    with wave.open(file_path, 'rb') as wave_file:
        sample_rate = wave_file.getframerate()
    return sample_rate

# 사용 예시
file_path = './dereverb/0bTCIbyvBBc.wav'
sr = get_wave_file_sample_rate(file_path)
print(f"Sample rate of {file_path}: {sr} Hz")