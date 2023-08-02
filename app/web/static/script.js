document.addEventListener("DOMContentLoaded", () => {
  let lastUploadedImage = null;
  const dropzone = document.getElementById("image-dropzone");
  const featureSelect = document.getElementById("feature-select");
  const thinkingText = document.getElementById("thinking-text");

  function handleImageUpload(file) {
    const predictedImagesContainer = document.getElementById("predicted-images");
    predictedImagesContainer.innerHTML = "";
    const selectedFeature = featureSelect.value;
    const formData = new FormData();
    thinkingText.innerText = "One sec, thinking...";
    formData.append("uploaded-image", file);
    formData.append("feature", selectedFeature);
    lastUploadedImage = file;
    fetch("/guess", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        displayResults(data);
      })
      .catch((error) => {
        console.error("Error uploading image:", error);
      });

    const reader = new FileReader();
    reader.onloadend = () => {
      const uploadedImage = new Image();
      uploadedImage.src = reader.result;
      uploadedImage.classList.add("uploaded-image");
      uploadedImage.onload = () => {
        const dropzoneWidth = dropzone.offsetWidth;
        const dropzoneHeight = dropzone.offsetHeight;
        const imageWidth = uploadedImage.width;
        const imageHeight = uploadedImage.height;
        const widthScale = dropzoneWidth / imageWidth;
        const heightScale = dropzoneHeight / imageHeight;
        const scaleFactor = Math.min(widthScale, heightScale);
        uploadedImage.width = imageWidth * (scaleFactor-0.1);
        uploadedImage.height = imageHeight * (scaleFactor-0.1);
        dropzone.innerHTML = "";
        dropzone.appendChild(uploadedImage);
      };
    };

    if (file) {
      reader.readAsDataURL(file);
    }
  }


dropzone.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      handleImageUpload(file);
    };
    fileInput.click();
});


dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("highlight");
});


dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("highlight");
});


dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("highlight");

    const file = e.dataTransfer.files[0];
    handleImageUpload(file);
});

featureSelect.addEventListener("change", () => {
  if (lastUploadedImage) {
    handleImageUpload(lastUploadedImage);
  }
});

function displayResults(data) {
  thinkingText.innerText = "Looks like...";
  const predictedImagesContainer = document.getElementById("predicted-images");
  predictedImagesContainer.innerHTML = ""; // Clear previous images

  for (let i = 0; i < data.images.length; i++) {
    const imageUrl = data.images[i];
    const predictionUrl = data.predictions[i];
    const name = data.names[i];

    // Create a container div for each image and its anchor
    const imageContainer = document.createElement("div");

    // Create the anchor element
    const anchorElement = document.createElement("a");
    anchorElement.href = predictionUrl;
    anchorElement.target = "_blank"; // Open the link in a new tab

    // Create the image element
    const imageElement = document.createElement("img");
    // Add cache-busting parameter to the image URL
    const cacheBustUrl = imageUrl + `?cache=${Date.now()}`;
    imageElement.src = cacheBustUrl;

    const tooltip = document.createElement("div");
    tooltip.classList.add("image-tooltip");
    tooltip.textContent = name; // Use the name from the 'names' list as the tooltip text

    // Append the image to the anchor and the anchor to the container
    anchorElement.appendChild(imageElement);
    imageContainer.appendChild(tooltip);
    imageContainer.appendChild(anchorElement);

    // Append the container to the predictedImagesContainer
    predictedImagesContainer.appendChild(imageContainer);


    // Add the tooltip element to the document body
    document.body.appendChild(tooltip);

    // Add event listeners to handle tooltip visibility
    imageContainer.addEventListener("mouseover", () => {
      tooltip.style.opacity = "1";
    });

    imageContainer.addEventListener("mouseout", () => {
      tooltip.style.opacity = "0";
    });

    // Position the tooltip dynamically based on cursor movement
    imageContainer.addEventListener("mousemove", (event) => {
      tooltip.style.left = event.pageX + "px";
      tooltip.style.top = event.pageY + "px";
    });
  }
}


});
