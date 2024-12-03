<script>
  import clsx from 'clsx'
  import SvelteMarkdown from 'svelte-markdown'
  import { nodes, selectedNode, postPrompt } from './data'
  import { tick } from 'svelte'

  let convEl
  let waiting = false

  $: n = $selectedNode || ($nodes && $nodes[0])
  $: conversation = n
    ? n.generation
        .reduce((conv, gen) => {
          conv.push(...[gen.prompt, gen.answer])
          return conv
        }, [])
        .map((msg) => msg.replace(/[\s"']+$/, '').replace(/^[\s"']+/, ''))
    : []

  function prompt(txt) {
    conversation = [...conversation, txt]
    waiting = true
    postPrompt($selectedNode.id, txt)
      .then((r) => {
        conversation = [...conversation, r.response]
      })
      .finally(() => {
        waiting = false
      })
  }

  $: if (conversation && waiting) {
    tick().then(() => scrollToBottom())
  }

  function scrollToBottom() {
    convEl.scrollTop = convEl.scrollHeight
  }
</script>

<div class="flex h-full w-full flex-col">
  <div class="h-full w-full overflow-y-auto" bind:this={convEl}>
    {#each conversation as msg, i}
      <div
        class={clsx(
          'm-2 flex flex-col gap-2 rounded-lg p-3 text-gray-100 shadow-lg',
          i % 2 !== 0
            ? 'mr-10 rounded-tl-none bg-zinc-700'
            : 'ml-10 rounded-tr-none bg-sky-800'
        )}
      >
        <SvelteMarkdown source={msg} />
      </div>
    {/each}
    {#if waiting}
      <div class="m-2 flex flex-row justify-end">
        <div class="rounded-lg rounded-tr-none bg-sky-800 px-5 py-3">
          {#each [0, 1, 2] as i}
            <span class="font-icon text-[8px]">circle</span>&nbsp;
          {/each}
        </div>
      </div>
    {/if}
  </div>
  <div
    class="flex items-center justify-center border-y-[1px] border-y-gray-700 p-1"
  >
    <div class="material-icons font-icon cursor-default text-xl text-gray-400">
      message
    </div>
    <input
      disabled={$selectedNode === null}
      on:focus={scrollToBottom}
      on:keypress={(e) => {
        if (e.key === 'Enter') {
          prompt(e.currentTarget.value)
          e.currentTarget.value = ''
        }
      }}
      type="text"
      class="w-full border-solid border-gray-100 bg-transparent p-2 text-lg
        placeholder-gray-600 !outline-none"
      placeholder={$selectedNode === null
        ? 'Select a node to continue the conversation'
        : '...'}
    />
  </div>
</div>
