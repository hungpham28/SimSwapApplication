(function ($) {
    'use strict';
    $(function () {
        $('.file-upload-browse').on('click', function () {
            var file = $(this).parent().parent().parent().find('.file-upload-default');
            file.trigger('click');
        });
        $('.file-upload-default').on('change', function () {
            $(this).parent().find('.form-control').val($(this).val().replace(/C:\\fakepath\\/i, ''));
        });
    });
})(jQuery);

function previewVideo() {
    const input = document.getElementById('video-input');
    const videoPreview = document.getElementById('video-preview');

    const file = input.files[0];
    if (file.size > 50 * 1024 * 1024) {
        alert("Video file is too big!");
        input.value = "";
    } else if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            videoPreview.src = e.target.result;
            videoPreview.style.display = 'block';
        };

        reader.readAsDataURL(file);
    }
}