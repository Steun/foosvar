// register left and right keys with Vue
Vue.config.keyCodes.left = 37;
Vue.config.keyCodes.right = 39;

var app = new Vue({
  el: "#app",
  data: {
    message: "Hello Vue!",
    timePercentage: 0,
    frameMs: 0.03,
    playbackSpeed: 0.5
  },
  methods: {
    startTimer() {
      setInterval(() => {
        const video = document.querySelector("#video");
        video.playbackRate = this.playbackSpeed;
        const maxTime = video.duration;
        this.timePercentage = video.currentTime;
      }, 100);
    },
    handleSliderChange(e) {
      const video = document.querySelector("#video");
      video.pause();
      video.currentTime = this.timePercentage;
      this.timePercentage = e.srcElement.value;
    },
    play() {
      document.querySelector("#video").play();
    },
    togglePlay() {
      const video = document.querySelector("#video");
      video.paused ? video.play() : video.pause();
    },
    goFrame(modifier) {
      const video = document.querySelector("#video");
      video.currentTime = video.currentTime + this.frameMs * modifier;
    },
    setPlaybackSpeed(value) {
      this.playbackSpeed = value;
      document.querySelector("#video").playbackRate = value;
    }
  },
  mounted: function() {
    this.startTimer();
  }
});
