document.addEventListener("DOMContentLoaded", function () {
    document.querySelector("button[type='submit']").addEventListener("click", function (event) {
        event.preventDefault(); // Prevent form submission

        const textInput = document.getElementById("text").value.trim();
        const imageInput = document.getElementById("image").files[0];
        const resultDiv = document.getElementById("result");

        // Clear previous messages
        resultDiv.innerHTML = "";

        // Validate inputs
        if (!textInput) {
            resultDiv.innerHTML = "<p>⚠️ Please enter some text.</p>";
            return;
        }
        if (!imageInput) {
            resultDiv.innerHTML = "<p>⚠️ Please upload an image.</p>";
            return;
        }

        // Display image name for confirmation
        console.log(`Image selected: ${imageInput.name}`);

        // Prepare form data
        let formData = new FormData();
        formData.append("text", textInput);
        formData.append("image", imageInput);

        console.log("Sending request to Flask server...");

        // Send request to Flask server
        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Response received from server:", data);

            if (data.error) {
                resultDiv.innerHTML = `<p style="color: red;">❌ Error: ${data.error}</p>`;
            } else {
                resultDiv.innerHTML = `
                    <p style="font-size: 20px;">${data.label}</p>
                    <p><b>Final Score:</b> ${data.final_score}%</p>
                    <p><b>Text Analysis Score:</b> ${data.text_score}%</p>
                    <p><b>Image Analysis Score:</b> ${data.image_score}%</p>
                `;
            }
        })
        .catch(error => {
            console.error("Fetch error:", error);
            resultDiv.innerHTML = "<p style='color: red;'>❌ Error connecting to the server. Make sure the Flask backend is running!</p>";
        });
    });
});
