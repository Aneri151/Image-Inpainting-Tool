document.addEventListener('DOMContentLoaded', function(){
  // Index page: drag/drop and AJAX upload with progress
  const dropArea = document.querySelector('.upload-area');
  const fileInput = document.querySelector('input[type="file"][name="image"]');
  const previewImg = document.getElementById('preview-img');
  const form = document.getElementById('upload-form');
  const progressBar = document.getElementById('upload-progress');
  const progressContainer = document.getElementById('progress-container');

  if(dropArea){
    ['dragenter','dragover'].forEach(evt=>dropArea.addEventListener(evt, e=>{e.preventDefault();e.stopPropagation();dropArea.classList.add('dragover')}));
    ['dragleave','drop'].forEach(evt=>dropArea.addEventListener(evt, e=>{e.preventDefault();e.stopPropagation();dropArea.classList.remove('dragover')}));
    dropArea.addEventListener('drop', e=>{const f = e.dataTransfer.files[0]; if(f){fileInput.files = e.dataTransfer.files; showPreview(f);}});
    dropArea.addEventListener('click', ()=>fileInput.click());
    fileInput.addEventListener('change', ()=>{ showPreview(fileInput.files[0]); });
  }

  function showPreview(file){
    if(!file) return; const url = URL.createObjectURL(file); previewImg.src = url; previewImg.style.display='block';
  }

  if(form){
    form.addEventListener('submit', function(){
      const file = fileInput.files[0];
      if(!file){
        alert('Please select an image');
        return false;
      }
      progressContainer.style.display = 'block';
      progressBar.style.width = '10%';
      progressBar.innerText = 'Uploading...';
      return true;
    });
  }

  // Result page: compare slider
  const range = document.getElementById('compare-range');
  if(range){
    const overlay = document.querySelector('.compare-overlay');
    const handleUpdate = ()=>{ const pct = range.value; overlay.style.width = pct + '%'; };
    range.addEventListener('input', handleUpdate); handleUpdate();
  }

  // Thumbnail modal
  const modal = document.getElementById('imageModal');
  if(modal){
    const modalImg = modal.querySelector('.modal-img');
    const closeButton = modal.querySelector('.modal-close');

    function openModal(url){
      modalImg.src = url;
      modal.classList.add('show');
      modal.setAttribute('aria-hidden', 'false');
    }

    function closeModal(){
      modal.classList.remove('show');
      modal.setAttribute('aria-hidden', 'true');
      modalImg.src = '';
    }

    document.querySelectorAll('.thumb-click').forEach(el => {
      el.addEventListener('click', e => {
        openModal(e.currentTarget.dataset.src);
      });
    });

    closeButton.addEventListener('click', closeModal);
    modal.addEventListener('click', e => {
      if(e.target === modal){
        closeModal();
      }
    });

    document.addEventListener('keydown', e => {
      if(e.key === 'Escape' && modal.classList.contains('show')){
        closeModal();
      }
    });
  }
});
