### Gait-Recognition-YOLO-v8

Gait-Recognition-YOLO-v8 is a project for gait recognition using YOLO version 8. This document provides basic setup instructions to get you started.

#### Prerequisites

Make sure you have the following installed on your system:

- Python (>=3.6)
- Pip (Python package installer)

#### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Kartik-Katkar/Gait-Recognition-YOLO-v8.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Gait-Recognition-YOLO-v8
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. If you wish to Download a pretrained model, you can download it from [Download Model](https://github.com/jackhanyuan/GaitRecognitionSystem/releases/download/1.1/output.zip) make sure to unzip the compressed file to `model/gait/output`

#### Usage

Run the gait recognition script:

```bash
python3 main.py
```

Feel free to customize the instructions based on the specific requirements of your project.
