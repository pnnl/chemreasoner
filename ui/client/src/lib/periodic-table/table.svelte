<script>
  import { elements } from './table.json'
  import Element from './element.svelte'

  import { nodes, selectedNode } from '../data'

  let el
  let w
  let h

  $: node = $selectedNode
  $: elems = node
    ? node.symbols.reduce((acc, symbols) => {
        symbols.symbols.forEach((symbol) => {
          symbol.forEach((s) => {
            acc.set(s, acc.has(s) ? acc.get(s) + 1 : 1)
          })
        })
        return acc
      }, new Map())
    : $nodes
      ? $nodes.reduce((acc, node) => {
          node.symbols.forEach((symbols) => {
            symbols.symbols.forEach((symbol) => {
              symbol.forEach((s) => {
                acc.set(s, acc.has(s) ? acc.get(s) + 1 : 1)
              })
            })
          })
          return acc
        }, new Map())
      : new Map()
  $: maxCount = Array.from(elems.values()).reduce(
    (acc, count) => Math.max(acc, count),
    -Infinity
  )

  $: if (el) {
    new ResizeObserver(() => {
      const { width, height } = el.getBoundingClientRect()
      w = width / 18
      h = height / 10
    }).observe(el)
  }
</script>

<div class="table h-full w-full text-gray-900" bind:this={el}>
  {#each elements as elem}
    <Element
      atomicNumber={elem.number}
      style="grid-column: {elem.xpos}; grid-row: {elem.ypos}"
      count={elems.get(elem.symbol, 0)}
      {maxCount}
      width={w}
      height={h}
    />
  {:else}{null}{/each}
</div>

<style>
  .table {
    display: grid;
    grid-template-columns: repeat(18, 1fr);
    grid-template-rows: repeat(10, 1fr);
    gap: 3px;
  }
</style>
