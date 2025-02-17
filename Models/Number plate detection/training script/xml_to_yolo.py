import os
import xml.etree.ElementTree as ET

# Define paths
xml_folder = "C:/Users/arjun/Downloads/License Plate dataset/annotations"  # Change this
txt_folder = "C:/Users/arjun/Downloads/License Plate dataset/labels"  # Change this
classes = ["licence"]  # Update with your class names

if not os.path.exists(txt_folder):
    os.makedirs(txt_folder)

# Function to convert XML to YOLO format
def convert_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    txt_filename = os.path.join(txt_folder, image_name.replace(".png", ".txt"))

    with open(txt_filename, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in classes:
                continue

            class_id = classes.index(class_name)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to YOLO format (normalized)
            x_center = (xmin + xmax) / (2.0 * image_width)
            y_center = (ymin + ymax) / (2.0 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Process all XML files
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(xml_folder, xml_file))

print("Conversion Completed!")
