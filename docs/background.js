fetch("https://api.codetabs.com/v1/proxy/?quest=http://scann.ddns.net:8000/health").then((res)=>{
    let status = document.getElementById("status")
    if(res.ok) 
    {
        console.log("200")
        status.textContent = "Online"
        status.style.color = "green"
        status.style.cursor = "pointer"
    }else
    {
        console.log("uh oh")
        status.textContent = "Offline"
        status.style.cursor = "pointer"
        status.addEventListener('mouseover',()=>{
            status.textContent = "Contact me "
            status.style.color = "white"
            status.addEventListener('mousedown',()=>{
                window.open("mailto:minh.mangbachkhoahochiminh@hcmut.edu.vn",'minh.mangbachkhoahochiminh@hcmut.edu.vn').focus()
            })
        })
    }
})