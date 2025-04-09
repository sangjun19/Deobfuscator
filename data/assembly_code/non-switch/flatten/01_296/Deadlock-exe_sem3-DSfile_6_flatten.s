	.file	"Deadlock-exe_sem3-DSfile_6_flatten.c"
	.text
	.globl	_TIG_IZ_3sEt_argc
	.bss
	.align 4
	.type	_TIG_IZ_3sEt_argc, @object
	.size	_TIG_IZ_3sEt_argc, 4
_TIG_IZ_3sEt_argc:
	.zero	4
	.globl	_TIG_IZ_3sEt_envp
	.align 8
	.type	_TIG_IZ_3sEt_envp, @object
	.size	_TIG_IZ_3sEt_envp, 8
_TIG_IZ_3sEt_envp:
	.zero	8
	.globl	_TIG_IZ_3sEt_argv
	.align 8
	.type	_TIG_IZ_3sEt_argv, @object
	.size	_TIG_IZ_3sEt_argv, 8
_TIG_IZ_3sEt_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"enter matrix a :"
.LC1:
	.string	"enter matrix b :"
.LC2:
	.string	"matrix A"
.LC3:
	.string	"\nsparse matrix as1 : "
.LC4:
	.string	"\nsparse matrix as2 : "
	.align 8
.LC5:
	.string	"\nbefore displaying sparse matirx"
	.align 8
.LC6:
	.string	"after displaying sparse matirx"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$0, _TIG_IZ_3sEt_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_3sEt_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_3sEt_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 128 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3sEt--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_3sEt_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_3sEt_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_3sEt_envp(%rip)
	nop
	movq	$0, -56(%rbp)
.L11:
	cmpq	$2, -56(%rbp)
	je	.L6
	cmpq	$2, -56(%rbp)
	ja	.L13
	cmpq	$0, -56(%rbp)
	je	.L8
	cmpq	$1, -56(%rbp)
	jne	.L13
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	movl	$2, -64(%rbp)
	movl	$2, -60(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %ecx
	movq	-48(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	read_array
	movq	%rax, -48(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %ecx
	movq	-40(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	read_array
	movq	%rax, -40(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %ecx
	movq	-48(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	display_array
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %ecx
	movq	-40(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	display_array
	movq	$0, -24(%rbp)
	movq	$0, -16(%rbp)
	movl	-60(%rbp), %ecx
	movl	-64(%rbp), %edx
	movq	-24(%rbp), %rsi
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	conv_2D_to_sparse
	movq	%rax, -24(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	display_sparse_matrix
	movl	-60(%rbp), %ecx
	movl	-64(%rbp), %edx
	movq	-16(%rbp), %rsi
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	conv_2D_to_sparse
	movq	%rax, -16(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	display_sparse_matrix
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	add_sparsematrix
	movq	%rax, -32(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	display_sparse_matrix
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	movq	-8(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	conv_sparse_to_normal
	movq	%rax, -8(%rbp)
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %ecx
	movq	-8(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	display_array
	movq	$2, -56(%rbp)
	jmp	.L9
.L8:
	movq	$1, -56(%rbp)
	jmp	.L9
.L6:
	movl	$0, %eax
	jmp	.L12
.L13:
	nop
.L9:
	jmp	.L11
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC7:
	.string	"inside while loop %d\n"
	.align 8
.LC8:
	.string	"Cannot add matrix of different order"
	.align 8
.LC9:
	.string	"\nentered in else block of add sparse matrix"
	.align 8
.LC10:
	.string	"\ndisplay inside add of sparse matrix : "
	.text
	.globl	add_sparsematrix
	.type	add_sparsematrix, @function
add_sparsematrix:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -88(%rbp)
	movq	%rsi, -96(%rbp)
	movq	$14, -32(%rbp)
.L60:
	cmpq	$31, -32(%rbp)
	ja	.L61
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L39-.L17
	.long	.L61-.L17
	.long	.L38-.L17
	.long	.L61-.L17
	.long	.L37-.L17
	.long	.L61-.L17
	.long	.L36-.L17
	.long	.L61-.L17
	.long	.L35-.L17
	.long	.L34-.L17
	.long	.L33-.L17
	.long	.L32-.L17
	.long	.L61-.L17
	.long	.L31-.L17
	.long	.L30-.L17
	.long	.L29-.L17
	.long	.L28-.L17
	.long	.L61-.L17
	.long	.L27-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L24-.L17
	.long	.L23-.L17
	.long	.L22-.L17
	.long	.L21-.L17
	.long	.L61-.L17
	.long	.L61-.L17
	.long	.L20-.L17
	.long	.L19-.L17
	.long	.L18-.L17
	.long	.L61-.L17
	.long	.L16-.L17
	.text
.L27:
	movq	-40(%rbp), %rax
	jmp	.L40
.L37:
	movl	$0, %eax
	jmp	.L40
.L30:
	movq	$21, -32(%rbp)
	jmp	.L41
.L29:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -32(%rbp)
	jmp	.L41
.L16:
	movl	-48(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jg	.L42
	movq	$15, -32(%rbp)
	jmp	.L41
.L42:
	movq	$20, -32(%rbp)
	jmp	.L41
.L35:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rcx
	movq	-96(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L44
	movq	$29, -32(%rbp)
	jmp	.L41
.L44:
	movq	$23, -32(%rbp)
	jmp	.L41
.L22:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %edx
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rcx
	movq	-96(%rbp), %rax
	addq	%rcx, %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jge	.L46
	movq	$6, -32(%rbp)
	jmp	.L41
.L46:
	movq	$10, -32(%rbp)
	jmp	.L41
.L28:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rcx
	movq	-96(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L48
	movq	$28, -32(%rbp)
	jmp	.L41
.L48:
	movq	$8, -32(%rbp)
	jmp	.L41
.L21:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -32(%rbp)
	jmp	.L41
.L24:
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -68(%rbp)
	movq	-88(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -64(%rbp)
	movq	-96(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -60(%rbp)
	movq	-96(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -56(%rbp)
	movq	$0, -40(%rbp)
	movl	$4, %edi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$1, -52(%rbp)
	movl	$1, -48(%rbp)
	movl	$1, -44(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L41
.L32:
	movl	-64(%rbp), %eax
	cmpl	-56(%rbp), %eax
	je	.L50
	movq	$24, -32(%rbp)
	jmp	.L41
.L50:
	movq	$27, -32(%rbp)
	jmp	.L41
.L34:
	movl	-68(%rbp), %eax
	cmpl	-60(%rbp), %eax
	je	.L52
	movq	$11, -32(%rbp)
	jmp	.L41
.L52:
	movq	$27, -32(%rbp)
	jmp	.L41
.L31:
	call	create_sparse_element
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-68(%rbp), %edx
	movl	%edx, (%rax)
	movq	-8(%rbp), %rax
	movl	-64(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-8(%rbp), %rax
	movl	$0, 8(%rax)
	movq	$2, -32(%rbp)
	jmp	.L41
.L26:
	movl	-52(%rbp), %eax
	cmpl	-68(%rbp), %eax
	jg	.L54
	movq	$31, -32(%rbp)
	jmp	.L41
.L54:
	movq	$20, -32(%rbp)
	jmp	.L41
.L36:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	4(%rax), %eax
	movl	%eax, 4(%rdx)
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	8(%rax), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -52(%rbp)
	addl	$1, -44(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L41
.L20:
	cmpq	$0, -40(%rbp)
	jne	.L56
	movq	$13, -32(%rbp)
	jmp	.L41
.L56:
	movq	$2, -32(%rbp)
	jmp	.L41
.L23:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	4(%rax), %eax
	movl	%eax, 4(%rdx)
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	8(%rax), %ecx
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	8(%rax), %edx
	movl	-44(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rsi
	movq	-40(%rbp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, 8(%rax)
	addl	$1, -52(%rbp)
	addl	$1, -44(%rbp)
	addl	$1, -48(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L41
.L19:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	4(%rax), %eax
	movl	%eax, 4(%rdx)
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	8(%rax), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -52(%rbp)
	addl	$1, -44(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L41
.L33:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %edx
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rcx
	movq	-96(%rbp), %rax
	addq	%rcx, %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jle	.L58
	movq	$0, -32(%rbp)
	jmp	.L41
.L58:
	movq	$22, -32(%rbp)
	jmp	.L41
.L39:
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	4(%rax), %eax
	movl	%eax, 4(%rdx)
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	8(%rax), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -48(%rbp)
	addl	$1, -44(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L41
.L18:
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	4(%rax), %eax
	movl	%eax, 4(%rdx)
	movl	-48(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-44(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$5, %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	8(%rax), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -48(%rbp)
	addl	$1, -44(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L41
.L38:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -32(%rbp)
	jmp	.L41
.L25:
	movq	-40(%rbp), %rax
	movl	-68(%rbp), %edx
	movl	%edx, (%rax)
	movq	-40(%rbp), %rax
	movl	-56(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-40(%rbp), %rax
	movl	-44(%rbp), %edx
	movl	%edx, 8(%rax)
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	display_sparse_matrix
	movq	$18, -32(%rbp)
	jmp	.L41
.L61:
	nop
.L41:
	jmp	.L60
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	add_sparsematrix, .-add_sparsematrix
	.section	.rodata
.LC11:
	.string	"%3d "
.LC12:
	.string	"normal matrix form : "
.LC13:
	.string	"[ "
.LC14:
	.string	"]"
	.text
	.globl	display_array
	.type	display_array, @function
display_array:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$2, -8(%rbp)
.L79:
	cmpq	$13, -8(%rbp)
	ja	.L80
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L65(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L65(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L65:
	.long	.L80-.L65
	.long	.L72-.L65
	.long	.L71-.L65
	.long	.L80-.L65
	.long	.L80-.L65
	.long	.L80-.L65
	.long	.L80-.L65
	.long	.L70-.L65
	.long	.L69-.L65
	.long	.L68-.L65
	.long	.L67-.L65
	.long	.L66-.L65
	.long	.L80-.L65
	.long	.L81-.L65
	.text
.L69:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L73
.L72:
	movl	-12(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jge	.L74
	movq	$8, -8(%rbp)
	jmp	.L73
.L74:
	movq	$7, -8(%rbp)
	jmp	.L73
.L66:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L73
.L68:
	movl	-16(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L76
	movq	$10, -8(%rbp)
	jmp	.L73
.L76:
	movq	$13, -8(%rbp)
	jmp	.L73
.L67:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L73
.L70:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L73
.L71:
	movq	$11, -8(%rbp)
	jmp	.L73
.L80:
	nop
.L73:
	jmp	.L79
.L81:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	display_array, .-display_array
	.globl	create_sparse_element
	.type	create_sparse_element, @function
create_sparse_element:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$1, -24(%rbp)
.L88:
	cmpq	$2, -24(%rbp)
	je	.L83
	cmpq	$2, -24(%rbp)
	ja	.L90
	cmpq	$0, -24(%rbp)
	je	.L85
	cmpq	$1, -24(%rbp)
	jne	.L90
	movq	$2, -24(%rbp)
	jmp	.L86
.L85:
	movq	-32(%rbp), %rax
	jmp	.L89
.L83:
	movl	$32, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$0, -8(%rbp)
	movq	-32(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 24(%rax)
	movq	-32(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$0, -24(%rbp)
	jmp	.L86
.L90:
	nop
.L86:
	jmp	.L88
.L89:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	create_sparse_element, .-create_sparse_element
	.globl	conv_sparse_to_normal
	.type	conv_sparse_to_normal, @function
conv_sparse_to_normal:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	$7, -24(%rbp)
.L122:
	cmpq	$23, -24(%rbp)
	ja	.L124
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L94(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L94(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L94:
	.long	.L124-.L94
	.long	.L109-.L94
	.long	.L108-.L94
	.long	.L107-.L94
	.long	.L106-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L105-.L94
	.long	.L104-.L94
	.long	.L103-.L94
	.long	.L124-.L94
	.long	.L102-.L94
	.long	.L101-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L100-.L94
	.long	.L99-.L94
	.long	.L98-.L94
	.long	.L124-.L94
	.long	.L97-.L94
	.long	.L96-.L94
	.long	.L124-.L94
	.long	.L95-.L94
	.long	.L93-.L94
	.text
.L106:
	movq	-56(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jge	.L110
	movq	$20, -24(%rbp)
	jmp	.L112
.L110:
	movq	$9, -24(%rbp)
	jmp	.L112
.L100:
	movl	$0, -32(%rbp)
	movq	$22, -24(%rbp)
	jmp	.L112
.L101:
	movl	$0, -28(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L112
.L104:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$3, -24(%rbp)
	jmp	.L112
.L109:
	movl	$1, -40(%rbp)
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -36(%rbp)
	movq	$23, -24(%rbp)
	jmp	.L112
.L93:
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L113
	movq	$11, -24(%rbp)
	jmp	.L112
.L113:
	movq	$15, -24(%rbp)
	jmp	.L112
.L107:
	addl	$1, -28(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L112
.L99:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$3, -24(%rbp)
	jmp	.L112
.L102:
	movq	-56(%rbp), %rax
	movl	4(%rax), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movq	-16(%rbp), %rax
	movq	%rax, (%rdx)
	addl	$1, -36(%rbp)
	movq	$23, -24(%rbp)
	jmp	.L112
.L103:
	addl	$1, -32(%rbp)
	movq	$22, -24(%rbp)
	jmp	.L112
.L97:
	movq	-64(%rbp), %rax
	jmp	.L123
.L98:
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	movq	(%rdx), %rcx
	movl	-28(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rcx, %rdx
	movl	8(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -40(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L112
.L95:
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jge	.L116
	movq	$12, -24(%rbp)
	jmp	.L112
.L116:
	movq	$19, -24(%rbp)
	jmp	.L112
.L105:
	movq	$1, -24(%rbp)
	jmp	.L112
.L108:
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L118
	movq	$17, -24(%rbp)
	jmp	.L112
.L118:
	movq	$8, -24(%rbp)
	jmp	.L112
.L96:
	movl	-40(%rbp), %eax
	cltq
	salq	$5, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jne	.L120
	movq	$2, -24(%rbp)
	jmp	.L112
.L120:
	movq	$16, -24(%rbp)
	jmp	.L112
.L124:
	nop
.L112:
	jmp	.L122
.L123:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	conv_sparse_to_normal, .-conv_sparse_to_normal
	.globl	conv_2D_to_sparse
	.type	conv_2D_to_sparse, @function
conv_2D_to_sparse:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movl	%edx, -68(%rbp)
	movl	%ecx, -72(%rbp)
	movq	$3, -16(%rbp)
.L155:
	cmpq	$21, -16(%rbp)
	ja	.L157
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L128(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L128(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L128:
	.long	.L142-.L128
	.long	.L157-.L128
	.long	.L141-.L128
	.long	.L140-.L128
	.long	.L139-.L128
	.long	.L138-.L128
	.long	.L157-.L128
	.long	.L157-.L128
	.long	.L137-.L128
	.long	.L136-.L128
	.long	.L135-.L128
	.long	.L157-.L128
	.long	.L157-.L128
	.long	.L157-.L128
	.long	.L134-.L128
	.long	.L133-.L128
	.long	.L132-.L128
	.long	.L131-.L128
	.long	.L157-.L128
	.long	.L130-.L128
	.long	.L129-.L128
	.long	.L127-.L128
	.text
.L139:
	movl	$0, -28(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L143
.L134:
	movq	-64(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, 8(%rax)
	movq	$17, -16(%rbp)
	jmp	.L143
.L133:
	addl	$1, -32(%rbp)
	movq	$19, -16(%rbp)
	jmp	.L143
.L137:
	call	create_sparse_element
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-32(%rbp), %edx
	movl	%edx, (%rax)
	movq	-8(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, 4(%rax)
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-8(%rbp), %rax
	movl	%edx, 8(%rax)
	movq	-8(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 24(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	addl	$1, -36(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L143
.L140:
	movq	$0, -16(%rbp)
	jmp	.L143
.L132:
	call	create_sparse_element
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-68(%rbp), %edx
	movl	%edx, (%rax)
	movq	-8(%rbp), %rax
	movl	-72(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-8(%rbp), %rax
	movl	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L143
.L127:
	movl	-28(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jge	.L144
	movq	$2, -16(%rbp)
	jmp	.L143
.L144:
	movq	$15, -16(%rbp)
	jmp	.L143
.L136:
	cmpq	$0, -64(%rbp)
	je	.L146
	movq	$10, -16(%rbp)
	jmp	.L143
.L146:
	movq	$17, -16(%rbp)
	jmp	.L143
.L130:
	movl	-32(%rbp), %eax
	cmpl	-68(%rbp), %eax
	jge	.L148
	movq	$4, -16(%rbp)
	jmp	.L143
.L148:
	movq	$14, -16(%rbp)
	jmp	.L143
.L131:
	movq	-64(%rbp), %rax
	jmp	.L156
.L138:
	addl	$1, -28(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L143
.L135:
	movl	$0, -36(%rbp)
	movl	$0, -32(%rbp)
	movq	$19, -16(%rbp)
	jmp	.L143
.L142:
	movq	$0, -8(%rbp)
	movq	$0, -24(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L143
.L141:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L151
	movq	$8, -16(%rbp)
	jmp	.L143
.L151:
	movq	$5, -16(%rbp)
	jmp	.L143
.L129:
	cmpq	$0, -64(%rbp)
	jne	.L153
	movq	$16, -16(%rbp)
	jmp	.L143
.L153:
	movq	$9, -16(%rbp)
	jmp	.L143
.L157:
	nop
.L143:
	jmp	.L155
.L156:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	conv_2D_to_sparse, .-conv_2D_to_sparse
	.section	.rodata
.LC15:
	.string	"element [%d][%d] = "
.LC16:
	.string	" %d"
	.text
	.globl	read_array
	.type	read_array, @function
read_array:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movl	%edx, -64(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$7, -32(%rbp)
.L175:
	cmpq	$13, -32(%rbp)
	ja	.L178
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L161(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L161(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L161:
	.long	.L178-.L161
	.long	.L178-.L161
	.long	.L168-.L161
	.long	.L178-.L161
	.long	.L167-.L161
	.long	.L178-.L161
	.long	.L178-.L161
	.long	.L166-.L161
	.long	.L178-.L161
	.long	.L165-.L161
	.long	.L164-.L161
	.long	.L163-.L161
	.long	.L162-.L161
	.long	.L160-.L161
	.text
.L167:
	movq	-56(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L176
	jmp	.L177
.L162:
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, (%rdx)
	movl	$0, -36(%rbp)
	movq	$13, -32(%rbp)
	jmp	.L170
.L163:
	addl	$1, -40(%rbp)
	movq	$10, -32(%rbp)
	jmp	.L170
.L165:
	movl	-36(%rbp), %eax
	leal	1(%rax), %edx
	movl	-40(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -36(%rbp)
	movq	$13, -32(%rbp)
	jmp	.L170
.L160:
	movl	-36(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jge	.L171
	movq	$9, -32(%rbp)
	jmp	.L170
.L171:
	movq	$11, -32(%rbp)
	jmp	.L170
.L164:
	movl	-40(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L173
	movq	$12, -32(%rbp)
	jmp	.L170
.L173:
	movq	$4, -32(%rbp)
	jmp	.L170
.L166:
	movq	$2, -32(%rbp)
	jmp	.L170
.L168:
	movl	-60(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -40(%rbp)
	movq	$10, -32(%rbp)
	jmp	.L170
.L178:
	nop
.L170:
	jmp	.L175
.L177:
	call	__stack_chk_fail@PLT
.L176:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	read_array, .-read_array
	.section	.rodata
.LC17:
	.string	"sparse matrix form:"
.LC18:
	.string	"%4d %4d %4d\n"
	.text
	.globl	display_sparse_matrix
	.type	display_sparse_matrix, @function
display_sparse_matrix:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$4, -8(%rbp)
.L191:
	cmpq	$6, -8(%rbp)
	ja	.L192
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L182(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L182(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L182:
	.long	.L186-.L182
	.long	.L185-.L182
	.long	.L192-.L182
	.long	.L184-.L182
	.long	.L183-.L182
	.long	.L192-.L182
	.long	.L193-.L182
	.text
.L183:
	movq	$3, -8(%rbp)
	jmp	.L187
.L185:
	cmpq	$0, -16(%rbp)
	je	.L188
	movq	$0, -8(%rbp)
	jmp	.L187
.L188:
	movq	$6, -8(%rbp)
	jmp	.L187
.L184:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L187
.L186:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -28(%rbp)
	movq	-16(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -24(%rbp)
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %ecx
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L187
.L192:
	nop
.L187:
	jmp	.L191
.L193:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	display_sparse_matrix, .-display_sparse_matrix
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
