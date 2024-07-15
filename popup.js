chrome.tabs.query({ 'active': true, 'lastFocusedWindow': true }, function (tabs) {
    var url = tabs[0].url;

    // Send the URL to the backend
    fetch('http://localhost:5000/scrape', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        const extElement = document.getElementById('ext');
        if (extElement) {
            extElement.innerHTML = ''; 
            data.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.classList.add('item');

                const linkElement = document.createElement('a');
                linkElement.href = item.link;
                linkElement.textContent = item.title;
                linkElement.target = '_blank';

                itemDiv.appendChild(linkElement);

                extElement.appendChild(itemDiv);
            });
        }
    })
    .catch(error => console.error('Error:', error));
});
