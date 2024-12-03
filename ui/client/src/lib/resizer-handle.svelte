<script>
  import clsx from 'clsx'
  export let vertical = false
</script>

<div
  class={clsx(
    'middle flex h-2 min-w-full items-center justify-center gap-1 bg-gray-950',
    vertical && 'min-h-full w-2 flex-col'
  )}
>
  {#each Array.from({ length: 3 }) as _, i}
    <div class="r-full h-1 w-1 rounded-full bg-gray-600"></div>
  {/each}
</div>
