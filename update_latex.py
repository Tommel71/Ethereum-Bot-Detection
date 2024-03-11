import shutil

def run(prefix):
    var_folder = f"{prefix}/outputs/large/latex_snippets"
    var_output = f"{prefix}/../latex/variables/large"

    figure_folder = f"{prefix}/outputs/large/figures"
    figure_output = f"{prefix}/../latex/figures/large"

    powerpoints_folder = f"{prefix}/powerpoints/outputs"
    powerpoints_output = f"{prefix}/../latex/powerpoints"


    # delete directory even if it is not empty
    shutil.rmtree(var_output, ignore_errors=True)
    shutil.rmtree(figure_output, ignore_errors=True)
    shutil.rmtree(powerpoints_output, ignore_errors=True)


    shutil.copytree(var_folder, var_output)
    shutil.copytree(figure_folder, figure_output)
    shutil.copytree(powerpoints_folder, powerpoints_output)


    print("latex updated")

if __name__ == "__main__":
    prefix = "."
    run(prefix)