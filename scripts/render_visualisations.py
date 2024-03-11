import os
import importlib
import pandas as pd

def run(prefix):

    vis_folder = f"{prefix}/src/Visualisations"
    # all the classes are called Vis, and each py file contains one. Get them all, instantiate and call render_visualisation
    vis_files = [f for f in os.listdir(vis_folder) if f.endswith(".py")]
    # remove the __init__ file
    vis_files.remove("__init__.py")
    data_rendering = []
    for vis_file in vis_files:

        vis_name = vis_file.split(".")[0]

        try:

            name = f"src.Visualisations.{vis_name}"
            vis_class = getattr(importlib.import_module(name), "Vis")
            vis = vis_class()
            vis.prefix = prefix
            vis.render_visualisation()
            data_rendering.append({"name": vis_name, "status": "success"})
        except Exception as e:
            print(e)
            data_rendering.append({"name": vis_name, "status": "failed"})


    print(pd.DataFrame(data_rendering))





if __name__ == "__main__":
    prefix = ".."
    run(prefix)