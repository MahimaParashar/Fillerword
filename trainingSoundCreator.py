from utils import find_disfluency, umm_tracker, ctm_index_from_time
from pydub import AudioSegment
from pydub.playback import play
import moviepy
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import *
import os
from ffmpy import FFmpeg
import sys
import av
from pydub import AudioSegment
from pydub.playback import play
from tqdm import tqdm

import pdb; pdb.set_trace()

inp_dir="/media/hilab/HiLabData/Sagnik/filler_word_detection/allvideos/"
# out_dir="/media/hilab/HiLabData/Sagnik/filler_word_detection/TrainingDataGeneral/"
out_dir="/media/hilab/HiLabData/Sagnik/filler_word_detection/TrainingDataUmm/"

count = 1
for f in tqdm(os.listdir(inp_dir)):
    fname=os.path.splitext(f)[0]
    trans_ctm_file = None
    if f.endswith('.mp4'):
        ctm_file_path = "/media/hilab/HiLabData/Sagnik/filler_word_detection/data/ctm_dump/"+fname+"_transcript.ctm"
        out_file_path = "/media/hilab/HiLabData/Sagnik/filler_word_detection/data/out_dump/"+fname+"_out.txt"
        try:
            clip = VideoFileClip(inp_dir+f)
            clip.audio.write_audiofile(fname+"_audio.wav")
            clip.audio.write_audiofile("audio.wav")
            audio= AudioSegment.from_wav(fname+"_audio.wav")
            
            trans_ctm_file = open(ctm_file_path, "r")
            with open(out_file_path,"r") as _:
                pass
        except Exception as e:
            print("Down Sampling audio")
            os.system("ffmpeg -i audio.wav -y -ar 8000 test.wav")
            os.system("cp test.wav ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/input/1.wav")
            print("Decoding")
            os.system("./kaldi/egs/aspire/s5/steps/online/nnet3/decode.sh  --nj 1 --acwt 1.0 --post-decode-acwt 10.0 ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/transcription/ ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/decode")
            print("Best Path")
            os.system('./kaldi/src/latbin/lattice-best-path --lm-scale=12 --word-symbol-table=./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt "ark:zcat ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/decode/lat.1.gz |" ark,t:- | ./kaldi/egs/aspire/s5/utils/int2sym.pl  -f 2- ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt > transcript.txt')
            os.system("bash ./disfluencydetection/job2.sh ./transcript.txt "+out_file_path+"  punc notok")
            os.system('./kaldi/src/latbin/lattice-1best --lm-scale=12 "ark:zcat ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/decode/lat.1.gz |" ark:- | ./kaldi/src/latbin/lattice-align-words ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/phones/word_boundary.int ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/final.mdl ark:- ark:- |./kaldi/src/latbin/nbest-to-ctm ark:- - | ./kaldi/egs/aspire/s5/utils/int2sym.pl -f 5 ./kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt > ./data/ctm_dump/'+fname+'_transcript.ctm')

            trans_ctm_file = open(ctm_file_path, "r")

        wordlist = [[]]
        listidx = []
        for line in trans_ctm_file:
            wordlist.append(line.split(' '))

        wordlist = wordlist[1:] # the first element in empty
        wordlist = [word for word in wordlist if word[4][0] !='[']
        len_of_wordlist = len(wordlist)


        end_time_of_file = (float(wordlist[-1][2]) + float(wordlist[-1][3]))*3000

        time = 0
        start_window_time = float(wordlist[0][2]) * 3000

        while time <= end_time_of_file:
            end_window_time = start_window_time + 3000
            start_index = ctm_index_from_time(start_window_time,wordlist)
            end_index = ctm_index_from_time(end_window_time, wordlist, end_ind= True)

            # disfluencies = find_disfluency(start_window_time, start_index, end_index, ctm_file_path, out_file_path)
            disfluencies = umm_tracker(start_window_time, start_index, end_index, wordlist)
            
            # print("disfluencies",disfluencies)
            if not disfluencies:
                word_sound = audio[start_window_time:start_window_time + 3000]
                word_sound.export(out_dir + "segment_" + str(count)  + ".wav", format="wav")
                label_file = open(out_dir + "segment_" + str(count)  + "_label.txt", "a")
                label_file.write("")
                label_file.close()
            else:
                for j in range(len(disfluencies)):

                    start_time = float(disfluencies[j][0]) + float(start_window_time)  # pydub works in milliseconds
                    duration = (float(disfluencies[j][1]) - float(disfluencies[j][0]))
                    end_time = float(disfluencies[j][1])  + float(start_window_time)
                    window_start = end_time - 3000

                    window_start_list = []
                    if window_start < 0.0:
                        window_start = 0.0
                    interval = (start_time - window_start) / 4
                    if interval > 0:
                        for i in range(5):
                            window_start_list.append(window_start + interval * i)
                            word_sound = audio[window_start_list[i]:window_start_list[i] + 3000]
                            start_index = ctm_index_from_time(window_start_list[i], wordlist)
                            end_index = ctm_index_from_time(window_start_list[i]+3000, wordlist, end_ind=True)
                            
                            # disfluency = find_disfluency(window_start_list[i], start_index, end_index, ctm_file_path, out_file_path)
                            disfluency = umm_tracker(window_start_list[i], start_index, end_index, wordlist)
                            
                            if not disfluency:  
                                print("here", i) #something goes wrong on the fourth index
                            
                            
                            word_sound.export(out_dir + "segment_" + str(count) + "_" +str(j)+ str(i) + ".wav", format="wav")
                            
                            with open(out_dir + "segment_" + str(count) + "_" + str(j) + "_label.txt", "w") as label_file:
                                for k in range(len(disfluency)):
                                    label_file.write(str(disfluency[k][0]) + " " + str(disfluency[k][1]) + " " + str(disfluency[k][2]) + "\n")

            time = end_window_time
            start_window_time = 1500 + start_window_time
            count += 1
