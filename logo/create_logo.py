import os
import svgwrite
from svgwrite import cm, mm
from PIL import Image
import cairosvg
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg

def create_svg_logo(filename):
    dwg = svgwrite.Drawing(filename, profile='tiny', size=("500px", "200px"))

    # Background
    dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill="white"))

    # Define colors
    text_color = "black"    

    # Add text
    dwg.add(dwg.text("FraCSPy", insert=(50, 120), font_family="Poppins", font_size="100px", fill=text_color))    

    # Create magnifying glass handle for 'a'
    handle_coords = [(187, 143), (175, 112)]
    dwg.add(dwg.line(start=handle_coords[0], end=handle_coords[1], stroke=text_color, stroke_width=9))
       
    # Add curved stroke    
    stroke_path = "M155,90 Q155,78 165,78 Q157,81 155,90"
    dwg.add(dwg.path(stroke_path, stroke=text_color, stroke_width=0.2))
        
    dwg.save()


def convert_svg(svg_file, output_file, output_format, dpi=300, high_res_png=None):
    if output_format == 'eps':
        cairosvg.svg2eps(url=svg_file, write_to=output_file)
    else:
        # Set dimensions to match desired DPI with increased resolution for smoothness
        svg_width, svg_height = 500, 200  # Size defined in SVG creation
        scale_factor = dpi / 96.0  # SVG default DPI is 96
        upscale_factor = 2  # Increase resolution by a factor of 2 for smoothness
        
        if high_res_png is None:
            # Convert SVG to high-resolution PNG
            high_res_png = output_file.replace('.' + output_format, '_high_res.png')
            cairosvg.svg2png(url=svg_file, write_to=high_res_png, output_width=int(svg_width * scale_factor * upscale_factor), output_height=int(svg_height * scale_factor * upscale_factor))

        # Downscale image to target resolution with antialiasing
        img = Image.open(high_res_png)
        img = img.resize((int(svg_width * scale_factor), int(svg_height * scale_factor)), Image.LANCZOS)

        if output_format == 'png':
            img.save(output_file, dpi=(dpi, dpi))
        else:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output_file, dpi=(dpi, dpi))

    return high_res_png


def display_svg(svg_file):
    # Convert SVG to PNG for displaying
    png_data = cairosvg.svg2png(url=svg_file)
    img = mpimg.imread(BytesIO(png_data), format='png')

    # Plot the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Create output folder
output_folder = "./"
out_dpi = 300

# File paths
svg_file  = os.path.join(output_folder, "fracspy_logo.svg")
png_file  = os.path.join(output_folder, "fracspy_logo.png")
jpg_file  = os.path.join(output_folder, "fracspy_logo.jpg")
tiff_file = os.path.join(output_folder, "fracspy_logo.tiff")
eps_file  = os.path.join(output_folder, "fracspy_logo.eps")

# Create and convert logo
create_svg_logo(svg_file)
high_res_png = None  # Initialize high_res_png variable
high_res_png = convert_svg(svg_file, png_file, "png", dpi=out_dpi, high_res_png=high_res_png)
convert_svg(svg_file, jpg_file, "jpg", dpi=out_dpi, high_res_png=high_res_png)
convert_svg(svg_file, tiff_file, "tiff", dpi=out_dpi, high_res_png=high_res_png)
convert_svg(svg_file, eps_file, "eps", dpi=out_dpi, high_res_png=high_res_png)

# Display the SVG logo
display_svg(svg_file)
