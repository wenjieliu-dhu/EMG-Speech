const players = [...document.querySelectorAll("audio")];

for (const player of players) {
  player.addEventListener("play", () => {
    for (const otherPlayer of players) {
      if (otherPlayer !== player && !otherPlayer.paused) {
        otherPlayer.pause();
      }
      otherPlayer.classList.toggle("is-playing", otherPlayer === player);
    }
  });

  player.addEventListener("pause", () => {
    player.classList.remove("is-playing");
  });

  player.addEventListener("ended", () => {
    player.classList.remove("is-playing");
  });
}
