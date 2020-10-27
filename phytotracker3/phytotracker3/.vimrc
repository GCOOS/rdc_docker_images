syntax enable
filetype plugin indent on
set ruler
set expandtab
set showmatch
set matchtime=3
inoremap jk <ESC>
:nnoremap <silent> <F5> :let _s=@/<Bar>:%s/\s\+$//e<Bar>:let @/=_s<Bar>:nohl<CR>
let mapleader= ","
set term=builtin_ansi
set encoding=utf-8
autocmd FileType litcoffee runtime ftplugin/coffee.vim
autocmd BufNewFile,BufRead *.coffee set filetype=coffee
autocmd FileType html setlocal shiftwidth=2 softtabstop=2
autocmd FileType javascript setlocal shiftwidth=2 softtabstop=2
autocmd FileType coffee setlocal shiftwidth=2 softtabstop=2
autocmd FileType python setlocal shiftwidth=4 softtabstop=4
let python_highlight_all = 1
if has("autocmd")
    au BufReadPost * if line("'\"") > 0 && line("'\"") <= line("$")
    \| exe "normal! g'\"" | endif
endif
autocmd StdinReadPre * let s:std_in=1
autocmd BufWritePost *.coffee silent make! --bare
autocmd BufWritePre *.py :%s/\s\+$//e
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
set makeprg=python\ %
