import os
import importlib
import pandas as pd

def run(prefix):

    tab_folder = f"{prefix}/src/TablesPaper"
    # all the classes are called Vis, and each py file contains one. Get them all, instantiate and call render_visualisation
    tab_files = [f for f in os.listdir(tab_folder) if f.endswith(".py")]
    # remove the __init__ file
    tab_files.remove("__init__.py")
    data_rendering = []
    for tab_file in tab_files:

        tab_name = tab_file.split(".")[0]

        try:

            name = f"src.Tables.{tab_name}"
            table_class = getattr(importlib.import_module(name), "Tab")
            table = table_class()
            table.prefix = prefix
            table.create_and_save()
            data_rendering.append({"name": tab_name, "status": "success"})
        except Exception as e:
            print(e)
            data_rendering.append({"name": tab_name, "status": "failed"})


    print(pd.DataFrame(data_rendering))

if __name__ == "__main__":
    prefix = ".."
    run(prefix)