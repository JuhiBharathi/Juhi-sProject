function processText() {
    let text = document.getElementById("textInput").value.trim();
    let selectedTool = document.getElementById("toolSelector").value;
    let targetLang = document.getElementById("targetLang").value;
    let outputElement = document.getElementById("output");

    if (text === "") {
        outputElement.innerHTML = "⚠️ Please enter some text.";
        return;
    }

    let formData = new URLSearchParams();
    formData.append("text", text);
    formData.append("option", selectedTool);

    if (selectedTool === "translate_text") {
        formData.append("target_lang", targetLang);
    }

    fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (selectedTool === "title_generator") {
            let titleList = "<b>Suggested Titles:</b><ul>";
            data.result.forEach(title => {
                titleList += `<li>${title}</li>`;
            });
            titleList += "</ul>";
            outputElement.innerHTML = titleList;
        }
        else if (selectedTool === "blog_outline") {
            let outlineHTML = "<b>Blog Outline:</b><br><br>";
            outlineHTML += `<b>Introduction:</b> ${data.result.Introduction}<br><br>`;
            outlineHTML += `<b>Key Points:</b><ul>`;
            data.result["Key Points"].forEach(point => {
                outlineHTML += `<li>${point}</li>`;
            });
            outlineHTML += `</ul><br><b>Conclusion:</b> ${data.result.Conclusion}`;
            outputElement.innerHTML = outlineHTML;
        }
        else if (selectedTool === "synonym_suggestion") {
            let formattedSynonyms = "";
            for (let word in data.result) {
                formattedSynonyms += `<b>${word}:</b> ${data.result[word].join(", ")}<br>`;
            }
            outputElement.innerHTML = formattedSynonyms;
        }
        else if (selectedTool === "hashtag_generator") {
            // ✅ Properly format multiple hashtags
            if (data.result.length > 0) {
                let hashtags = data.result.map(tag => `<span class="hashtag">${tag}</span>`).join(" ");
                outputElement.innerHTML = `<b>Generated Hashtags:</b><br>${hashtags}`;
            } else {
                outputElement.innerHTML = `<b>No Hashtags Found</b>`;
            }
        }
        else {
            outputElement.innerHTML = data.result;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        outputElement.innerHTML = "⚠️ An error occurred while processing your request.";
    });
}

// ✅ Show/Hide Language Dropdown for Translation
document.getElementById("toolSelector").addEventListener("change", function() {
    let languageSelector = document.getElementById("languageSelector");
    languageSelector.style.display = (this.value === "translate_text") ? "block" : "none";
});
