import moviepy.video.io.ImageSequenceClip as Movie_maker
from Modules.params import get_params
from os import listdir as ls
from os.path import join
from sys import argv


def create_animation(params: dict,
                     name: str,
                     fps: int = 20):
    """
    Funcion que ejecuta la creacion de la animacion
    """
    path = join(params["path graphics"],
                name)
    filenames = sorted(ls(path))
    filenames = [join(path,
                      filename)
                 for filename in filenames]
    output_file = join(params["path graphics"],
                       name)
    output_file = f"{output_file}.mp4"
    movie = Movie_maker.ImageSequenceClip(filenames,
                                          fps=fps,)
    movie.write_videofile(output_file)
    # logger=None)


params = get_params()
create_animation(params,
                 argv[1])
