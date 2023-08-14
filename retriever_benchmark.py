import os
import torch

from src.options import get_options
import logging
from src import dist_utils, slurm, util
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.tasks import get_task

logger = logging.getLogger(__name__)

def main(
    model,
    index,
    passages,
    opt
):
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    task = get_task(opt, unwrapped_model.reader_tokenizer)
    step = 0
    while step < opt.total_steps:
        data_iterator = task.data_iterator(
            opt.train_data, opt.global_rank, opt.world_size, repeat_if_less_than_world_size=True, opt=opt
        )
        data_iterator = filter(None, map(task.process, data_iterator))
        data_iterator = task.batch_iterator(data_iterator, opt.per_gpu_batch_size, drop_last=True, shuffle=opt.shuffle)
    

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    
    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "retriever.log"))
    
    if opt.is_main:
        options.print_options(opt)
        
    logger.info(f"World size: {dist_utils.get_world_size()}")
    
    logger.info(f"Loading index and passages")
    index, passages = load_or_initialize_index(opt)
    
    # logger.info(f"Loading model and retrever")
    # model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step = load_or_initialize_atlas_model(opt)
    
    local_device = torch.device("cuda", opt.local_rank)
    occupied_memory = torch.tensor(torch.cuda.memory_allocated(local_device), dtype=torch.float64, device=local_device)
    dist_utils.sum_main(occupied_memory)
    
    logger.info(f"Memory occupied until now: {occupied_memory / 1024 ** 3:.2f} GB")
    
    dist_utils.barrier()
    dist_utils.destroy_process_group()