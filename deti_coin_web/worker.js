// worker.js
importScripts('deti_coin.js');

createModule().then(Module => {
    console.log("WASM ready in worker, starting miner...");
    Module._main();
});
