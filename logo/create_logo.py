import os
import re
import numpy as np
import svgwrite
from PIL import Image
import cairosvg
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg

def ricker_wavelet(t, a=1):
    """
    Generate a Ricker wavelet.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time array.
    a : :obj:`float`, optional
        Alpha parameter controlling the wavelet's shape. Default is 6.

    Returns
    -------
    :obj:`numpy.ndarray`
        The generated Ricker wavelet.
    """
    return (1 - 2*(a*t)**2) * np.exp(-a**2 * t**2)

def create_svg_icon(filename, theme="light", background="circle"):
    """
    Create an SVG icon for lens with wavelet.

    This function generates an SVG icon with a specified filename.

    Parameters
    ----------
    filename : :obj:`str`
        The name of the file to save the SVG icon as.
    theme : :obj:`str`, optional, default: "light"
        Color theme: "light" or "dark"
    background : :obj:`str`, optional, default: "circle"
        Icon background: "circle" or "square"
    
    Raises
    ------
    ValueError :
        If theme is not "light" or "dark"
    ValueError :
        If background is not "circle" or "square"
    """
    # Set up the SVG drawing object
    dwg = svgwrite.Drawing(filename, profile='tiny', size=("100px", "100px"))

    # Define colors depending on the theme
    if theme == "light":
        text_color = "black"
        background_color = "white"
    elif theme == "dark":
        text_color = "white"
        background_color = "black"
    else:
        raise ValueError(f"Unknown theme: {theme}")

    # Add background
    if background == "circle":
        # Add a white circular background
        dwg.add(dwg.circle(center=(50, 50), r=50, fill=background_color))
    elif background == "square":
        # Add a white square background
        dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=background_color))
    else:
        raise ValueError(f"Unknown background: {background}")
    
    
    # Set up the time range and number of points for generating the wavelet
    t_min = -4
    t_max = 4
    num_points = 200
    t = np.linspace(t_min, t_max, num_points)

    # Generate the Ricker wavelet and adjust its shape
    wavelet = ricker_wavelet(t)
    distorted_wavelet = wavelet * (max(t) - t) * 5 + 1.2*t
    y = 47 - distorted_wavelet
    t = t * 5 + 52
    
    # Convert the data to SVG path format and add it as a path in the drawing object
    wavelet_path = "M" + " ".join([f"{x},{y}" for x, y in zip(t, y)])
    dwg.add(dwg.path(wavelet_path, stroke=text_color, stroke_width=0.2, fill=text_color))
        
    # Draw an oval with variable thickness using paths
    num_segments = 100  # Number of segments to approximate the oval
    angle_step = 2 * np.pi / num_segments
    cx = 49  # Center x
    cy = 43  # Center y
    rx = 22  # X radius
    ry = 24  # Y radius

    for i in range(num_segments):
        angle_start = i * angle_step
        angle_end = (i + 1) * angle_step
        x1 = cx + rx * np.cos(angle_start)
        y1 = cy + ry * np.sin(angle_start)
        x2 = cx + rx * np.cos(angle_end)
        y2 = cy + ry * np.sin(angle_end)
        
        # Create a variable stroke width that changes with the segment index
        stroke_width = 7 - 2 * np.cos(angle_start)  # Adjust as necessary
        
        # Draw the segment with the calculated stroke width
        dwg.add(dwg.line(start=(x1, y1), end=(x2, y2), 
                         stroke=text_color, stroke_width=stroke_width, 
                         stroke_linecap="round"))

    # Create a magnifying glass handle for 'a' and add it as a line in the drawing object
    handle_coords = [(68, 94), (57, 66)]    
    dwg.add(dwg.line(start=handle_coords[0], end=handle_coords[1], stroke=text_color, stroke_width=9))

    # Add curved strokes to mimic lens flares and add them as paths in the drawing object    
    stroke_path = "M35,42 Q35,30 45,27 Q37,33 35,42"
    dwg.add(dwg.path(stroke_path, stroke=text_color, stroke_width=0.2, fill=text_color))

    # Save the SVG icon to a file
    dwg.save()

def create_svg_logo(filename, theme="light"):
    """
    Create an SVG logo for FraCSPy.

    This function generates an SVG logo with a specified filename.

    Parameters
    ----------
    filename : :obj:`str`
        The name of the file to save the SVG logo as.
    theme : :obj:`str`, optional, default: "light"
        Color theme: "light" or "dark"
    
    Raises
    ------
    ValueError :
        If theme is not "light" or "dark"
    """
    # Set up the SVG drawing object
    dwg = svgwrite.Drawing(filename, profile='tiny', size=("400px", "100px"))

     # Define colors depending on the theme
    if theme == "light":
        text_color = "black"
        background_color = "white"
    elif theme == "dark":
        text_color = "white"
        background_color = None
    else:
        raise ValueError(f"Unknown theme: {theme}")

    if background_color is not None:
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=("100%", "100%"), fill=background_color))

    # Set up the time range and number of points for generating the wavelet
    t_min = -4
    t_max = 4
    num_points = 200
    t = np.linspace(t_min, t_max, num_points)

    # Generate the Ricker wavelet and adjust its shape
    wavelet = ricker_wavelet(t)
    distorted_wavelet = wavelet * (max(t) - t) * 5 + 1.2*t
    y = 50 - distorted_wavelet
    t = t * 5 + 117
    
    # Convert the data to SVG path format and add it as a path in the drawing object
    wavelet_path = "M" + " ".join([f"{x},{y}" for x, y in zip(t, y)])
    dwg.add(dwg.path(wavelet_path, stroke=text_color, stroke_width=0.2, fill=text_color))

    # Add text "FraCSPy"
    dwg.add(dwg.text("FraCSPy", insert=(-5, 73), font_family="Poppins", font_size="100px", fill=text_color))

    # Create a magnifying glass handle for 'a' and add it as a line in the drawing object    
    handle_coords = [(133, 97), (122, 69)]
    dwg.add(dwg.line(start=handle_coords[0], end=handle_coords[1], stroke=text_color, stroke_width=9))

    # Add curved strokes to mimic lens flares and add them as paths in the drawing object    
    stroke_path = "M100,45 Q100,33 110,30 Q102,36 100,45"
    dwg.add(dwg.path(stroke_path, stroke=text_color, stroke_width=0.2, fill=text_color))

    # Save the SVG logo to a file
    dwg.save()

def convert_svg(svg_file, output_file, output_format, dpi=300, high_res_png=None, upscale_factor=2):
    """
    Convert SVG to target file format and resolution.

    Parameters
    ----------
    svg_file : :obj:`str`
        The name of the input SVG file.
    output_file : :obj:`str`
        The name of the output file.
    output_format : :obj:`str`
        The desired output file format (e.g., 'png', 'eps').
    dpi : :obj:`int`, optional, default is 300
        The target DPI for the converted image.
    high_res_png : :obj:`str`, optional, default is None
        The path to a high-resolution PNG file if it already exists.
        If None, this function will create one.
    upscale_factor : :obj:`int`, optional, default is 2
        Upscaling factor to create a high-resolution PNG file, valuable only if
        high_res_png is None

    Returns
    -------
    high_res_png : :obj:`str`
        The path to the generated high-resolution PNG file.
    """
    # Read SVG file content
    with open(svg_file, 'r') as f:
        svg_content = f.read()

    # Try to extract width and height from SVG content
    width_match = re.search(r'width="(\d+(\.\d+)?)(\w*)"', svg_content)
    height_match = re.search(r'height="(\d+(\.\d+)?)(\w*)"', svg_content)

    if width_match and height_match:
        svg_width = float(width_match.group(1))
        svg_height = float(height_match.group(1))
    else:
        # If width and height are not found, try to extract from viewBox
        viewbox_match = re.search(r'viewBox="([\d\s.]+)"', svg_content)
        if viewbox_match:
            viewbox = viewbox_match.group(1).split()
            svg_width = float(viewbox[2])
            svg_height = float(viewbox[3])
        else:
            # If still not found, use default values
            svg_width, svg_height = 400, 100

    if output_format == 'eps':
        cairosvg.svg2eps(url=svg_file, write_to=output_file)
    else:
        # Set dimensions to match desired DPI with increased resolution for smoothness        
        scale_factor = dpi / 96.0  # SVG default DPI is 96        

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
    """
    Display an SVG file as an image.

    Parameters
    ----------
    svg_file : :obj:`str`
        The name of the input SVG file.   
    """
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

# File paths for logo (light)
svg_logo_file  = os.path.join(output_folder, "fracspy_logo.svg")
png_logo_file  = os.path.join(output_folder, "fracspy_logo.png")
jpg_logo_file  = os.path.join(output_folder, "fracspy_logo.jpg")
tiff_logo_file = os.path.join(output_folder, "fracspy_logo.tiff")
eps_logo_file  = os.path.join(output_folder, "fracspy_logo.eps")
# File paths for logo (dark)
svg_logo_dark_file  = os.path.join(output_folder, "fracspy_logo_dark.svg")
png_logo_dark_file  = os.path.join(output_folder, "fracspy_logo_dark.png")
jpg_logo_dark_file  = os.path.join(output_folder, "fracspy_logo_dark.jpg")
tiff_logo_dark_file = os.path.join(output_folder, "fracspy_logo_dark.tiff")
eps_logo_dark_file  = os.path.join(output_folder, "fracspy_logo_dark.eps")
# File paths for icon
svg_icon_file  = os.path.join(output_folder, "fracspy_icon.svg")
png_icon_file  = os.path.join(output_folder, "fracspy_icon.png")
svg_icon_dark_file  = os.path.join(output_folder, "fracspy_icon_dark.svg")
png_icon_dark_file  = os.path.join(output_folder, "fracspy_icon_dark.png")

# Create logo
create_svg_logo(svg_logo_file)
create_svg_logo(svg_logo_dark_file,theme="dark")

# Create icon
create_svg_icon(svg_icon_file)
create_svg_icon(svg_icon_dark_file,theme="dark")

# Convert to PNG and upscaled PNG
upscale_factor = 2
high_res_logo_png = convert_svg(svg_logo_file, png_logo_file, "png", dpi=out_dpi, upscale_factor=upscale_factor)
high_res_logo_dark_png = convert_svg(svg_logo_dark_file, png_logo_dark_file, "png", dpi=out_dpi, upscale_factor=upscale_factor)
convert_svg(svg_icon_file, png_icon_file, "png", dpi=out_dpi, upscale_factor=upscale_factor)
convert_svg(svg_icon_dark_file, png_icon_dark_file, "png", dpi=out_dpi, upscale_factor=upscale_factor)

# Convert to other formats
convert_svg(svg_logo_file, jpg_logo_file, "jpg", dpi=out_dpi, high_res_png=high_res_logo_png)
convert_svg(svg_logo_file, tiff_logo_file, "tiff", dpi=out_dpi, high_res_png=high_res_logo_png)
convert_svg(svg_logo_file, eps_logo_file, "eps", dpi=out_dpi, high_res_png=high_res_logo_png)
convert_svg(svg_logo_dark_file, jpg_logo_dark_file, "jpg", dpi=out_dpi, high_res_png=high_res_logo_dark_png)
convert_svg(svg_logo_dark_file, tiff_logo_dark_file, "tiff", dpi=out_dpi, high_res_png=high_res_logo_dark_png)
convert_svg(svg_logo_dark_file, eps_logo_dark_file, "eps", dpi=out_dpi, high_res_png=high_res_logo_dark_png)

# Display the SVG icon and logo
display_svg(svg_logo_file)
display_svg(svg_icon_file)
