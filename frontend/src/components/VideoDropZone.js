const handleDrop = (event) => {

  event.preventDefault();

  const files = event.dataTransfer.files;

  const newMedia = [];

  for (let i = 0; i < files.length; i++) {

    const file = files[i];
    const url = URL.createObjectURL(file);

    if (file.type.startsWith("video/")) {

      newMedia.push({
        name: file.name,
        url: url,
        type: "video"
      });

    }

    else if (file.type.startsWith("image/")) {

      newMedia.push({
        name: file.name,
        url: url,
        type: "image"
      });

    }

  }

  setMediaList(prev => [...prev, ...newMedia]);

  if (newMedia.length > 0) {
    setCurrentMedia(newMedia[0]);
  }

};