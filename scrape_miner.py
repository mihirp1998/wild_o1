import os
import ipdb
import argparse
st = ipdb.set_trace


from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe



def process_pdf(pdf_path, model_list=None, out_folder="final_out"):
    """Process a PDF file and generate markdown output with extracted images.
    
    Args:
        pdf_path (str): Path to the input PDF file
        model_list (list, optional): List of models to use. Defaults to empty list.
        out_folder (str, optional): Output folder name. Defaults to "final_out".
    """
    if model_list is None:
        model_list = []
        
    # Get PDF filename without extension
    pdf_filename = pdf_path.split("/")[-1][:-4]

    # Prepare output directories
    local_image_dir, local_md_dir = f"{out_folder}/images", f"{out_folder}" 
    os.makedirs(local_image_dir, exist_ok=True)

    # Initialize writers and readers
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)
    image_dir = str(os.path.basename(local_image_dir))
    
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_path)   # read the pdf content

    # Process PDF through pipeline
    pipe = OCRPipe(pdf_bytes, model_list, image_writer)
    pipe.pipe_classify()
    pipe.pipe_analyze() 
    pipe.pipe_parse()

    pdf_info = pipe.pdf_mid_data["pdf_info"]

    # Generate markdown output
    md_content = pipe.pipe_mk_markdown(
        image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
    )
    
    # Write markdown output
    md_filepath = f"{pdf_filename}.md"
    if isinstance(md_content, list):
        md_writer.write_string(md_filepath, "\n".join(md_content))
    else:
        md_writer.write_string(md_filepath, md_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF and generate markdown output with extracted images')
    parser.add_argument('--root_folder', type=str, required=False, default="/grogu/user/mprabhud/papers_o1/attention_papers",
                      help='Path to input PDF file')
    parser.add_argument('--start_idx', type=int, required=False, default=None,
                      help='Start index')
    parser.add_argument('--end_idx', type=int, required=False, default=None,
                      help='End index')
    parser.add_argument('--pdf', type=str, required=False, default="",
                      help='Path to input PDF file')
    parser.add_argument('--models', nargs='+', default=[],
                      help='List of models to use (default: [])')
    parser.add_argument('--out_folder', type=str, default='final_out',
                      help='Output folder name (default: final_out)')
    
    args = parser.parse_args()
    print(args)
    
    if args.pdf != "":
        process_pdf(args.pdf, args.models, args.out_folder)
    else:
        assert args.start_idx is not None and args.end_idx is not None, "start_idx and end_idx are required"
        for i in range(args.start_idx, args.end_idx):
            folder_name = f"{args.root_folder}/{i:05d}"
            pdf_path = f"{folder_name}/main.pdf"
            md_path = f"{folder_name}/main.md"
            print(f"Processing {i}")
            if not os.path.exists(pdf_path):
                print(f"Skipping {pdf_path} because it does not exist")
                continue
            
            if os.path.exists(md_path):
                print(f"Skipping {pdf_path} because {md_path} exists")
                continue
            process_pdf(pdf_path, args.models, folder_name)
            print(f"Processed {pdf_path}")
