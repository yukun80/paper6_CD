// 只运行所选组；每次改此项重新运行。assets仅只读检查已经完成的small阶段资产。
var PROBE_GROUP = 'quick'; // quick | geometry | topology | prepare | solver | gradient | assets
var PROBE_ASSETS = {assetRoot: '', runId: 'cfdepth_v321_small', step: 2};
// small资产必须由主入口fixture:'small'、其余默认数值参数生成，不在本探针创建任务。
