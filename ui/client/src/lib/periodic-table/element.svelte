<script>
  import clsx from 'clsx'
  import { elements } from './table.json'
  import { selectedElement } from '../data'

  export let style
  export let atomicNumber
  export let width
  export let height
  export let count = 0
  export let maxCount = 0

  let opacity = 0
  let hovering = false

  $: scaleFactor = Math.min(Math.min(width, height) / 60, 1)

  $: opacity = count > 0 ? count / maxCount : 0
</script>

<!-- <div
  class="element relative"
  style={`${style} --bg-opacity:${opacity}`}
  class:hl={count > 0}
> -->
<button
  type="button"
  class={clsx(
    'element cursor-default overflow-hidden rounded text-gray-500',
    // elements[atomicNumber - 1].category.replace(/ /g, '-'),
    opacity > 0 && opacity < 0.5 && 'text-white'
  )}
  class:hl={count > 0}
  style={`${style}; --bg-opacity:${opacity};`}
  title={`${elements[atomicNumber - 1].symbol}: ${elements[atomicNumber - 1].name} (${atomicNumber})`}
>
  <div
    class="element-atomic-number w-full text-left font-light"
    style:transform={`scale(${scaleFactor})`}
  >
    {atomicNumber}
  </div>
  <div
    class="element-symbol mx-auto text-center font-semibold"
    style:transform={`scale(${scaleFactor})`}
  >
    {elements[atomicNumber - 1].symbol}
  </div>
  <div
    class="element-name text-center font-light"
    style:transform={`scale(${scaleFactor})`}
  >
    {elements[atomicNumber - 1].name}
  </div>
</button>

<!-- {#if hovering || $selectedElement === elements[atomicNumber - 1]}
    <div
      class="absolute left-[calc(50%-5px)] top-[-5px] h-[10px] w-[10px] rounded-full
        bg-sky-500"
    ></div>
  {/if} -->
<!-- </div> -->

<style>
  .element {
    /* height: 4.8vh; */
    /* width: 4.8vw; */
    /* background-color: #cbd5e0; */
    /* background-color: rgba(255, 255, 255, 0.01); */
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }
  .element-atomic-number {
    font-size: 0.6vw;
    position: absolute;
    top: 2px;
    left: 3px;
    transform-origin: top left;
  }
  .element-symbol {
    font-size: 1.2vw;
    position: absolute;
    top: 0;
    bottom: 0;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    transform-origin: center;
  }
  .element-name {
    font-size: 0.6vw;
    position: absolute;
    bottom: 0;
    width: 100%;
    transform-origin: bottom;
  }
  .hl {
    --bg-opacity: 1;
    background-color: rgb(253, 224, 71, var(--bg-opacity));
  }
  .hl.diatomic-nonmetal {
    background-color: #faf089;
  }
  .hl.noble-gas {
    background-color: #fbd38d;
  }
  .hl.alkali-metal {
    background-color: #feb2b2;
  }
  .hl.alkaline-earth-metal {
    background-color: #7f9cf5;
  }
  .hl.metalloid {
    background-color: #90cdf4;
  }
  .hl.polyatomic-nonmetal {
    background-color: #81e6d9;
  }
  .hl.transition-metal {
    background-color: #9ae6b4;
  }
  .hl.post-transition-metal {
    background-color: #9ae6b4;
  }
  .hl.lanthanide {
    background-color: #fbb6ce;
  }
  .hl.actinide {
    background-color: #d6bcfa;
  }
</style>
