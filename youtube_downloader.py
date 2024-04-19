import requests
from dotenv import load_dotenv
import os
import time

env_path = ".env"

load_dotenv(dotenv_path=env_path)

from pytube import YouTube
from moviepy.editor import VideoFileClip


def get_youtube_links(query, max_results=500):
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "videoDuration": "short",  # video less then 5 minutes
        "key": os.environ.get("YOUTUBE_API_KEY"),
    }

    video_links = []

    while len(video_links) < max_results:
        # Make API request
        response = requests.get(base_url, params=params)
        data = response.json()

        # Extract video links
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            video_links.append(video_link)

            if len(video_links) >= max_results:
                break

        # Check for the presence of nextPageToken and update the params for the next page
        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
        else:
            break  # No more pages, exit the loop

    return video_links[:max_results]


def download_and_convert_videos(video_links, output_folder):
    for i, link in enumerate(video_links):
        try:
            # Download YouTube video
            yt = YouTube(link)
            video_stream = yt.streams.filter(file_extension="mp4", res="360p").first()
            video_stream.download(f"{output_folder}/mp4", filename=f"video_{i}.mp4")

            # Convert downloaded video to WAV
            video_path = f"{output_folder}/mp4/video_{i}.mp4"
            audio_path = f"{output_folder}/wav/audio_{i}.wav"

            video_clip = VideoFileClip(video_path)
            video_clip.audio.write_audiofile(
                audio_path, codec="pcm_s16le", ffmpeg_params=["-ac", "1"]
            )

            print(f"Downloaded and converted video {i}.")

        except Exception as e:
            print(f"Error downloading video {i}: {e}")


if __name__ == "__main__":
    try:
        print("initiating links")
        zouk_music_links = get_youtube_links("zouk music dance")

        output_folder = "zouk_music"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        download_and_convert_videos(zouk_music_links, output_folder)

        print("its all done!")
    except Exception as error:
        print(error)
