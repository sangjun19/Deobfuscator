	.file	"mohammadakeeb0011_Dsa_3rd_Sem_main_flatten.c"
	.text
	.globl	_TIG_IZ_NO2q_argc
	.bss
	.align 4
	.type	_TIG_IZ_NO2q_argc, @object
	.size	_TIG_IZ_NO2q_argc, 4
_TIG_IZ_NO2q_argc:
	.zero	4
	.globl	_TIG_IZ_NO2q_argv
	.align 8
	.type	_TIG_IZ_NO2q_argv, @object
	.size	_TIG_IZ_NO2q_argv, 8
_TIG_IZ_NO2q_argv:
	.zero	8
	.globl	_TIG_IZ_NO2q_envp
	.align 8
	.type	_TIG_IZ_NO2q_envp, @object
	.size	_TIG_IZ_NO2q_envp, 8
_TIG_IZ_NO2q_envp:
	.zero	8
	.text
	.globl	create
	.type	create, @function
create:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	$0, -16(%rbp)
.L7:
	cmpq	$2, -16(%rbp)
	je	.L8
	cmpq	$2, -16(%rbp)
	ja	.L9
	cmpq	$0, -16(%rbp)
	je	.L4
	cmpq	$1, -16(%rbp)
	jne	.L9
	movq	-40(%rbp), %rax
	movl	-44(%rbp), %edx
	movl	%edx, (%rax)
	movl	$-1, -20(%rbp)
	movq	-40(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-40(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, 8(%rax)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$2, -16(%rbp)
	jmp	.L5
.L4:
	movq	$1, -16(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	create, .-create
	.globl	isfull
	.type	isfull, @function
isfull:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L18:
	cmpq	$2, -8(%rbp)
	je	.L11
	cmpq	$2, -8(%rbp)
	ja	.L19
	cmpq	$0, -8(%rbp)
	je	.L13
	cmpq	$1, -8(%rbp)
	jne	.L19
	movl	24(%rbp), %eax
	movl	16(%rbp), %edx
	subl	$1, %edx
	cmpl	%edx, %eax
	jne	.L14
	movq	$2, -8(%rbp)
	jmp	.L16
.L14:
	movq	$0, -8(%rbp)
	jmp	.L16
.L13:
	movl	$0, %eax
	jmp	.L17
.L11:
	movl	$1, %eax
	jmp	.L17
.L19:
	nop
.L16:
	jmp	.L18
.L17:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	isfull, .-isfull
	.section	.rodata
.LC0:
	.string	"ERROR: Queue is empty!"
	.text
	.globl	dequeue
	.type	dequeue, @function
dequeue:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L32:
	cmpq	$5, -8(%rbp)
	ja	.L34
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L34-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L24:
	movq	-24(%rbp), %rax
	movl	4(%rax), %edx
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jne	.L28
	movq	$1, -8(%rbp)
	jmp	.L30
.L28:
	movq	$0, -8(%rbp)
	jmp	.L30
.L26:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L30
.L25:
	movl	$-1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L30
.L22:
	movl	-12(%rbp), %eax
	jmp	.L33
.L27:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 4(%rax)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L30
.L34:
	nop
.L30:
	jmp	.L32
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	dequeue, .-dequeue
	.section	.rodata
.LC1:
	.string	"The first element is %d\n"
	.align 8
.LC2:
	.string	"The element 40 is at index %d\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_NO2q_envp(%rip)
	nop
.L36:
	movq	$0, _TIG_IZ_NO2q_argv(%rip)
	nop
.L37:
	movl	$0, _TIG_IZ_NO2q_argc(%rip)
	nop
	nop
.L38:
.L39:
#APP
# 83 "mohammadakeeb0011_Dsa_3rd_Sem_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-NO2q--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_NO2q_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_NO2q_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_NO2q_envp(%rip)
	nop
	movq	$0, -40(%rbp)
.L45:
	cmpq	$2, -40(%rbp)
	je	.L40
	cmpq	$2, -40(%rbp)
	ja	.L48
	cmpq	$0, -40(%rbp)
	je	.L42
	cmpq	$1, -40(%rbp)
	jne	.L48
	leaq	-32(%rbp), %rax
	movl	$6, %esi
	movq	%rax, %rdi
	call	create
	leaq	-32(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	enqueue
	leaq	-32(%rbp), %rax
	movl	$20, %esi
	movq	%rax, %rdi
	call	enqueue
	leaq	-32(%rbp), %rax
	movl	$30, %esi
	movq	%rax, %rdi
	call	enqueue
	leaq	-32(%rbp), %rax
	movl	$40, %esi
	movq	%rax, %rdi
	call	enqueue
	leaq	-32(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	enqueue
	subq	$8, %rsp
	pushq	-16(%rbp)
	pushq	-24(%rbp)
	pushq	-32(%rbp)
	call	display
	addq	$32, %rsp
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	dequeue
	movl	%eax, -48(%rbp)
	movl	-48(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subq	$8, %rsp
	pushq	-16(%rbp)
	pushq	-24(%rbp)
	pushq	-32(%rbp)
	movl	$40, %edi
	call	search
	addq	$32, %rsp
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subq	$8, %rsp
	pushq	-16(%rbp)
	pushq	-24(%rbp)
	pushq	-32(%rbp)
	call	display
	addq	$32, %rsp
	movq	$2, -40(%rbp)
	jmp	.L43
.L42:
	movq	$1, -40(%rbp)
	jmp	.L43
.L40:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L46
	jmp	.L47
.L48:
	nop
.L43:
	jmp	.L45
.L47:
	call	__stack_chk_fail@PLT
.L46:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.globl	search
	.type	search, @function
search:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$2, -8(%rbp)
.L64:
	cmpq	$7, -8(%rbp)
	ja	.L65
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L52(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L52(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L52:
	.long	.L65-.L52
	.long	.L65-.L52
	.long	.L57-.L52
	.long	.L56-.L52
	.long	.L55-.L52
	.long	.L54-.L52
	.long	.L53-.L52
	.long	.L51-.L52
	.text
.L55:
	movl	$-1, %eax
	jmp	.L58
.L56:
	movl	-12(%rbp), %eax
	jmp	.L58
.L53:
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L59
.L54:
	movl	24(%rbp), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L60
	movq	$7, -8(%rbp)
	jmp	.L59
.L60:
	movq	$4, -8(%rbp)
	jmp	.L59
.L51:
	movq	32(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jne	.L62
	movq	$3, -8(%rbp)
	jmp	.L59
.L62:
	movq	$6, -8(%rbp)
	jmp	.L59
.L57:
	movl	20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L59
.L65:
	nop
.L59:
	jmp	.L64
.L58:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	search, .-search
	.globl	isempty
	.type	isempty, @function
isempty:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L74:
	cmpq	$2, -8(%rbp)
	je	.L67
	cmpq	$2, -8(%rbp)
	ja	.L75
	cmpq	$0, -8(%rbp)
	je	.L69
	cmpq	$1, -8(%rbp)
	jne	.L75
	movl	$1, %eax
	jmp	.L70
.L69:
	movl	24(%rbp), %edx
	movl	20(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L71
	movq	$1, -8(%rbp)
	jmp	.L73
.L71:
	movq	$2, -8(%rbp)
	jmp	.L73
.L67:
	movl	$0, %eax
	jmp	.L70
.L75:
	nop
.L73:
	jmp	.L74
.L70:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	isempty, .-isempty
	.section	.rodata
.LC3:
	.string	"%d "
	.text
	.globl	display
	.type	display, @function
display:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L88:
	cmpq	$6, -8(%rbp)
	ja	.L89
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L79(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L79(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L79:
	.long	.L83-.L79
	.long	.L89-.L79
	.long	.L90-.L79
	.long	.L81-.L79
	.long	.L89-.L79
	.long	.L80-.L79
	.long	.L78-.L79
	.text
.L81:
	movl	24(%rbp), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L84
	movq	$5, -8(%rbp)
	jmp	.L86
.L84:
	movq	$6, -8(%rbp)
	jmp	.L86
.L78:
	movl	$10, %edi
	call	putchar@PLT
	movq	$2, -8(%rbp)
	jmp	.L86
.L80:
	movq	32(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L86
.L83:
	movl	20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L86
.L89:
	nop
.L86:
	jmp	.L88
.L90:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	display, .-display
	.section	.rodata
.LC4:
	.string	"ERROR: Queue is full!"
	.text
	.globl	enqueue
	.type	enqueue, @function
enqueue:
.LFB10:
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
	movq	$1, -8(%rbp)
.L100:
	cmpq	$3, -8(%rbp)
	je	.L92
	cmpq	$3, -8(%rbp)
	ja	.L101
	cmpq	$2, -8(%rbp)
	je	.L94
	cmpq	$2, -8(%rbp)
	ja	.L101
	cmpq	$0, -8(%rbp)
	je	.L102
	cmpq	$1, -8(%rbp)
	jne	.L101
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	cmpl	%eax, %edx
	jne	.L96
	movq	$2, -8(%rbp)
	jmp	.L98
.L96:
	movq	$3, -8(%rbp)
	jmp	.L98
.L92:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$0, -8(%rbp)
	jmp	.L98
.L94:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L98
.L101:
	nop
.L98:
	jmp	.L100
.L102:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	enqueue, .-enqueue
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
