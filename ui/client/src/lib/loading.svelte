<script>
  import { fade } from 'svelte/transition'
  import clsx from 'clsx'

  export let size = '2xl'

  $: classes = clsx(
    'animate-spin rounded-full border-y-2 border-yellow-500',
    size === 'sm' && `h-4 w-4`,
    size === 'md' && `h-6 w-6`,
    size === 'lg' && `h-8 w-8`,
    size === 'xl' && `h-12 w-12`,
    size === '2xl' && `h-16 w-16`,
    size === '3xl' && `h-24 w-24`,
    size === '4xl' && `h-32 w-32`,
    size === '5xl' && `h-40 w-40`,
    size === '6xl' && `h-48 w-48`
  )
</script>

<div
  transition:fade
  class="absolute inset-0 z-50 flex items-center justify-center bg-gray-900 bg-opacity-90"
>
  <div class={classes}></div>
</div>
