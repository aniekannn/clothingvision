function toggleMenu(){
    const menu = document.querySelector(".menu-links");
    const icon = document.querySelector(".hamburger-icon");
    menu.classList.toggle("open")
    icon.classList.toggle("open")

}

document.addEventListener("DOMContentLoaded", () => {
  const name = "Aniekan Ekanem";
  const target = document.getElementById("typed-name");
  let index = 0;

  function type() {
    if (index < name.length) {
      target.textContent += name.charAt(index);
      index++;
      setTimeout(type, 150); // Typing speed
    }
  }

  type();
});