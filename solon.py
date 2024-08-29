import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import datetime
import os
import gradio as gr
GOOGLE_API_KEY = "AIzaSyBTEC44aXeJ3FtPlk8ZR3zyWEpaFLS1qVc"

# Plot of the story (Max 10 words)
# Duration of the story in minutes: 0.5, 1, 2
# Enter the genre of the story: horror, comedy, romance, sci-fi

def createStory(prompt, duration, genre):
    plot = prompt
    duration_in_mins = float(duration)
    number_of_words = float(duration_in_mins) * 150
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

    answer = llm(
        [
            SystemMessage(content="In a world of boundless creativity, you are an AI storyteller who weaves intriguing tales with depth. Join us on a journey where words come alive, characters thrive, and stories leave an everlasting impact. Let's create something magical together."),
            HumanMessage(content=
            f""" Write a story with the following plot : {plot}
                with the following tone : {genre}
                The story should be {number_of_words} words long.
                Instead of pronouns, use the names of the characters.
            """)
        ]
    )

    story = answer.content
    length_of_story = len(story.split(" "))
    with open("story.txt", "w") as file:
        file.write(story)
    print(f"Length of the story is {length_of_story}")

    def convert_to_list(string):
        words = string.split()
        result = [" ".join(words[i:i+15]) for i in range(0, len(words), 15)]
        return result

    word_lists = convert_to_list(story)
    print(word_lists)

    from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
    from langchain.prompts import PromptTemplate


    character_rail = f"""


    given the following story, please extract a list of characters and generate a detailed description of the character

    {story}

    """

    model = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY)
    # model = OpenAI(temperature=0.9 , openai_api_key=OPENAI_API_KEY) # type: ignore
    # model = OpenAI(temperature=0)
    print(story)

    characters = model.invoke(character_rail)
    print(characters)

    chat = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

    image_description_list = []
    start_time = datetime.datetime.now()
    filename = start_time.strftime("%Y%m%d%H%M%S") + ".txt"
    with open(filename, "w") as file:
        for index, phrase in enumerate(word_lists):

            answer = chat(
                [
                    SystemMessage(
                        content="You are an AI which is straight to the point and tell out 4 to 5 word description of the image."
                    ),
                    HumanMessage(
                        content=f"""
                        section: {phrase}
                        and here are the characters: {characters}
                        
                        Here is what you have to do
                        -dont use names, Never use names of the characters because the image generator will not understand them
                        -Describe the characters, their emotional feeling in that particular time and their physical traits
                        -Make sure you explain the action and situation happening in the time along with the charachter description, have more weightage on the situation rather than the charachter
                        -give most importance to that section of that story and generate description only and only based on that section
                        -you should generate a 15 word description of the image which the image generator will understand
                        -dont use names at all
                        
                        here are some examples
                        section:Leo guided Sammy through the wilderness, sharing his wisdom and protecting him from danger. As gratitude
                        character: Sammy : squirrel , Leo : Lion
                        output description: Lion guided squirrel through the wilderness, sharing Lion's wisdom and protecting squirrel from danger. As gratitude
                        
                        always return a  word description of the image
                        """
                    )
                ]
            )

            image_description = answer.content
            print(image_description)
            file.write(str(image_description)+"\n")
            image_description_list.append(str(image_description))
            progress = (index + 1) / len(word_lists) * 100
            print(f"Progress for image generation prompts: {progress:.2f}%")

    from elevenlabs.client import ElevenLabs
    from elevenlabs import play, save
    client = ElevenLabs(
    api_key="277f00c7afe1746a93166b07f571df83", # Defaults to ELEVEN_API_KEY
    )

    audio = client.generate(
    text=f"{story}",
    voice="Rachel",
    model="eleven_multilingual_v2"
    )

    save(audio, "audio.mp3")

    from PIL import Image, ImageFont, ImageDraw

# adding subtitle

    def add_text_to_image(image_path, text):
        # Open the image
        image = Image.open(image_path)

        # Calculate the font size based on the image size
        width, height = image.size
        font_size = int(height * 0.06)  # Adjust this multiplier to control the font size

        # Load the font
        font = ImageFont.truetype('arial.ttf', font_size)

        # Estimate the text size by getting the bounding box
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Adjust the font size if the text is wider than the image
        while text_width > width:
            font_size -= 1
            font = ImageFont.truetype('arial.ttf', font_size)
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

        # Calculate the position of the text at the bottom center of the image
        text_x = (width - text_width) // 2
        text_y = height - text_height - int(height * 0.05)  # Adjust this multiplier to control the vertical position

        # Create a translucent background for the text
        background_color = (0, 0, 0, 128)  # RGBA value with alpha transparency
        text_background = Image.new('RGBA', (text_width, text_height), background_color)

        # Draw the text on the background
        draw = ImageDraw.Draw(text_background)
        draw.text((0, 0), text, font=font, fill=(255, 255, 255))

        # Paste the text background onto the image
        image.paste(text_background, (text_x, text_y), mask=text_background)

        # Save the resulting image
        image.save(image_path)
    
    from diffusechain import Automatic1111

    api = Automatic1111(baseurl='http://localhost:7860')

    seed = -1

    for index, image_generation_dict in enumerate(image_description_list):
        # print("Description:", image_generation_dict["description"])
        # print("Positive Prompts:", image_generation_dict["positive_prompt"])
        # print("Negative Prompts:", image_generation_dict["negative_prompts"])
        # print()
        print(image_generation_dict)
        
        result = api.txt2img(
            
            prompt=f"{image_generation_dict}, Architecture Photography, Wildlife Photography, Car Photography, Food Photography, Interior Photography, Landscape Photography, Hyperdetailed Photography, Cinematic, Movie Still, Mid Shot Photo, Full Body Photo, Skin Details",
            negative_prompt="(((text))),((color)),(shading),background,noise,dithering,gradient,detailed,out of frame,ugly,error,Illustration, watermark, Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck",
            save_images=True
        )
        
        seed = result.info["seed"]
        result.image.save(f"images/image_{index}.jpg")
        add_text_to_image(f"images/image_{index}.jpg", f'{word_lists[index]}')
    
    
    from moviepy.editor import ImageClip, concatenate_videoclips
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.editor import ImageSequenceClip, concatenate
    from moviepy.editor import TextClip
    from moviepy.editor import CompositeVideoClip

    def create_video(image_dir, audio_file, subtitle_list, output_file):
        # Load the sequence of images
        image_files = sorted(os.listdir(image_dir))

        # Calculate the duration for each image based on the audio duration
        audio = AudioFileClip(audio_file)
        duration_per_image = audio.duration / len(image_files)

        clips = []
        for i, img in enumerate(image_files):
            # Create an ImageClip for each image with the specified duration
            image_path = os.path.join(image_dir, img)
            image = ImageClip(image_path, duration=duration_per_image)

            clips.append(image)

        # Concatenate the clips to create the final video
        final_clip = concatenate_videoclips(clips)

        # Set the audio for the final clip
        final_clip = final_clip.set_audio(audio)

        # Set the frame rate for the final clip
        final_clip.fps = 24

        # Write the video file
        final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', fps=24)

    # Provide the image directory, audio file, and output file path
    # only give the
    project_dir = './'
    image_directory = f'{project_dir}/images'
    audio_file = f'{project_dir}/audio.mp3'
    output_file = f'{project_dir}/output_video_1.mp4'
    subtitle = word_lists


    # Call the create_video function
    create_video(image_directory, audio_file, subtitle, output_file)

    return f'{project_dir}/output_video_1.mp4'

solonVideo = gr.Interface(
    fn=createStory,
    inputs=[
        gr.Textbox(label="Plot of the story (Max 10 Words): "), 
        gr.Textbox(label="Duration of the video in minutes: 0.5, 1, 2"), 
        gr.Textbox(label="Genre of the video: sci-fi, horror, slice of life, adventure, action")],
    outputs=gr.Video()
)

solonVideo.launch()