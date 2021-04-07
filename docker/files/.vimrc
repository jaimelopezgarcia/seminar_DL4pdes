set encoding=utf-8

set nu

call plug#begin('~/.vim/plugged')


Plug 'tpope/vim-fugitive'

Plug 'vim-scripts/indentpython.vim'

Plug 'vim-syntastic/syntastic'

Plug 'scrooloose/nerdtree'


Plug 'jistr/vim-nerdtree-tabs'

Plug 'kien/ctrlp.vim'

Plug 'Lokaltog/powerline', {'rtp': 'powerline/bindings/vim/'}

call plug#end()

let python_highlight_all=1
syntax on


let NERDTreeIgnore=['\.pyc$', '\~$'] "ignore files in NERDTree


au BufNewFile,BufRead *.py
    \ set tabstop=4  |
    \ set softtabstop=4  |
    \ set shiftwidth=4 |
    \ set textwidth=79 |
    \ set expandtab |
    \ set autoindent |
    \ set fileformat=unix



au BufNewFile,BufRead *.js, *.html, *.css
    \ set tabstop=2 |
    \ set softtabstop=2 |
    \ set shiftwidth=2
	
