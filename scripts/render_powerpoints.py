import os
from pdfCropMargins import crop

def render_powerpoint(pptx_folder, outputfolder, pptx_file):
    pptx_path = f"{pptx_folder}/{pptx_file}"
    pdf_path = f"{outputfolder}/{pptx_file[:-5]}.pdf"
    old_pdf_path = pptx_path[:-5] + ".pdf"

    # delete old pdf if exists
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    command = f"ppt2pdf file {pptx_path}"
    os.system(command)

    # remove excess whitespace around pdf
    crop(["-o", pdf_path, "-p", "1", old_pdf_path])

    os.remove(old_pdf_path)

def render_powerpoints(prefix):
    pptx_folder = f"{prefix}/powerpoints"
    # get all pptx files from folder
    pptx_files = [f for f in os.listdir(pptx_folder) if f.endswith(".pptx")]
    outputfolder = f"{pptx_folder}/outputs"
    for pptx_file in pptx_files:
        render_powerpoint(pptx_folder, outputfolder, pptx_file)

if __name__ == "__main__":
    render_powerpoints("..")