	.file	"KritagyaPainuly_DSA_Practical_4_flatten.c"
	.text
	.globl	_TIG_IZ_EFNI_argv
	.bss
	.align 8
	.type	_TIG_IZ_EFNI_argv, @object
	.size	_TIG_IZ_EFNI_argv, 8
_TIG_IZ_EFNI_argv:
	.zero	8
	.globl	_TIG_IZ_EFNI_envp
	.align 8
	.type	_TIG_IZ_EFNI_envp, @object
	.size	_TIG_IZ_EFNI_envp, 8
_TIG_IZ_EFNI_envp:
	.zero	8
	.globl	_TIG_IZ_EFNI_argc
	.align 4
	.type	_TIG_IZ_EFNI_argc, @object
	.size	_TIG_IZ_EFNI_argc, 4
_TIG_IZ_EFNI_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d\n"
.LC1:
	.string	"Stack is empty"
	.text
	.globl	peek
	.type	peek, @function
peek:
.LFB1:
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
	movq	$0, -8(%rbp)
.L13:
	cmpq	$4, -8(%rbp)
	ja	.L14
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L8-.L4
	.long	.L15-.L4
	.long	.L15-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L9
.L5:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L9
.L8:
	cmpl	$-1, -28(%rbp)
	jne	.L11
	movq	$3, -8(%rbp)
	jmp	.L9
.L11:
	movq	$4, -8(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L13
.L15:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	peek, .-peek
	.section	.rodata
.LC2:
	.string	"Stack is full"
	.text
	.globl	push
	.type	push, @function
push:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$3, -8(%rbp)
.L28:
	cmpq	$5, -8(%rbp)
	ja	.L29
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L29-.L19
	.long	.L23-.L19
	.long	.L30-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L30-.L19
	.text
.L20:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L24
.L23:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, (%rax)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$5, -8(%rbp)
	jmp	.L24
.L21:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$9, %eax
	jne	.L25
	movq	$4, -8(%rbp)
	jmp	.L24
.L25:
	movq	$1, -8(%rbp)
	jmp	.L24
.L29:
	nop
.L24:
	jmp	.L28
.L30:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	push, .-push
	.section	.rodata
.LC3:
	.string	"%d has been poped\n"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$2, -8(%rbp)
.L43:
	cmpq	$5, -8(%rbp)
	ja	.L44
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L34(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L34(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L34:
	.long	.L38-.L34
	.long	.L37-.L34
	.long	.L36-.L34
	.long	.L44-.L34
	.long	.L45-.L34
	.long	.L45-.L34
	.text
.L37:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	$4, -8(%rbp)
	jmp	.L40
.L38:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L40
.L36:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L41
	movq	$0, -8(%rbp)
	jmp	.L40
.L41:
	movq	$1, -8(%rbp)
	jmp	.L40
.L44:
	nop
.L40:
	jmp	.L43
.L45:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	pop, .-pop
	.section	.rodata
	.align 8
.LC4:
	.string	"Enter 1 Push 2 Pop 3 Peek 4 Display 5 isempty 6 Exit"
.LC5:
	.string	"%d"
.LC6:
	.string	"\n-----Exiting-----"
.LC7:
	.string	"Enter the number"
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_EFNI_envp(%rip)
	nop
.L47:
	movq	$0, _TIG_IZ_EFNI_argv(%rip)
	nop
.L48:
	movl	$0, _TIG_IZ_EFNI_argc(%rip)
	nop
	nop
.L49:
.L50:
#APP
# 100 "KritagyaPainuly_DSA_Practical_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EFNI--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_EFNI_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_EFNI_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_EFNI_envp(%rip)
	nop
	movq	$13, -56(%rbp)
.L75:
	cmpq	$19, -56(%rbp)
	ja	.L78
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L53(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L53(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L53:
	.long	.L63-.L53
	.long	.L78-.L53
	.long	.L62-.L53
	.long	.L61-.L53
	.long	.L60-.L53
	.long	.L59-.L53
	.long	.L58-.L53
	.long	.L57-.L53
	.long	.L56-.L53
	.long	.L78-.L53
	.long	.L78-.L53
	.long	.L78-.L53
	.long	.L78-.L53
	.long	.L55-.L53
	.long	.L54-.L53
	.long	.L78-.L53
	.long	.L78-.L53
	.long	.L78-.L53
	.long	.L78-.L53
	.long	.L52-.L53
	.text
.L60:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-68(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -56(%rbp)
	jmp	.L64
.L54:
	movl	-64(%rbp), %edx
	leaq	-48(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	isempty
	movq	$4, -56(%rbp)
	jmp	.L64
.L56:
	movl	-68(%rbp), %eax
	cmpl	$6, %eax
	ja	.L65
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L67(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L67(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L67:
	.long	.L65-.L67
	.long	.L72-.L67
	.long	.L71-.L67
	.long	.L70-.L67
	.long	.L69-.L67
	.long	.L68-.L67
	.long	.L66-.L67
	.text
.L66:
	movq	$19, -56(%rbp)
	jmp	.L73
.L68:
	movq	$14, -56(%rbp)
	jmp	.L73
.L69:
	movq	$7, -56(%rbp)
	jmp	.L73
.L70:
	movq	$2, -56(%rbp)
	jmp	.L73
.L71:
	movq	$3, -56(%rbp)
	jmp	.L73
.L72:
	movq	$6, -56(%rbp)
	jmp	.L73
.L65:
	movq	$5, -56(%rbp)
	nop
.L73:
	jmp	.L64
.L61:
	leaq	-64(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	pop
	movq	$4, -56(%rbp)
	jmp	.L64
.L55:
	movl	$-1, -64(%rbp)
	movq	$4, -56(%rbp)
	jmp	.L64
.L52:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -56(%rbp)
	jmp	.L64
.L58:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-60(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-60(%rbp), %ecx
	leaq	-64(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	push
	movq	$4, -56(%rbp)
	jmp	.L64
.L59:
	movq	$4, -56(%rbp)
	jmp	.L64
.L63:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L76
	jmp	.L77
.L57:
	movl	-64(%rbp), %edx
	leaq	-48(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	display
	movq	$4, -56(%rbp)
	jmp	.L64
.L62:
	movl	-64(%rbp), %edx
	leaq	-48(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	peek
	movq	$4, -56(%rbp)
	jmp	.L64
.L78:
	nop
.L64:
	jmp	.L75
.L77:
	call	__stack_chk_fail@PLT
.L76:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.section	.rodata
.LC8:
	.string	"%d "
	.text
	.globl	display
	.type	display, @function
display:
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
	movq	$4, -8(%rbp)
.L96:
	cmpq	$10, -8(%rbp)
	ja	.L97
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L82(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L82(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L82:
	.long	.L89-.L82
	.long	.L88-.L82
	.long	.L98-.L82
	.long	.L97-.L82
	.long	.L86-.L82
	.long	.L98-.L82
	.long	.L97-.L82
	.long	.L84-.L82
	.long	.L97-.L82
	.long	.L83-.L82
	.long	.L81-.L82
	.text
.L86:
	cmpl	$-1, -28(%rbp)
	jne	.L90
	movq	$0, -8(%rbp)
	jmp	.L92
.L90:
	movq	$7, -8(%rbp)
	jmp	.L92
.L88:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L92
.L83:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jg	.L93
	movq	$1, -8(%rbp)
	jmp	.L92
.L93:
	movq	$10, -8(%rbp)
	jmp	.L92
.L81:
	movl	$10, %edi
	call	putchar@PLT
	movq	$5, -8(%rbp)
	jmp	.L92
.L89:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L92
.L84:
	movl	$0, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L92
.L97:
	nop
.L92:
	jmp	.L96
.L98:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	display, .-display
	.section	.rodata
.LC9:
	.string	"Stack is not empty"
	.text
	.globl	isempty
	.type	isempty, @function
isempty:
.LFB11:
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
.L111:
	cmpq	$4, -8(%rbp)
	ja	.L112
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L102(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L102(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L102:
	.long	.L113-.L102
	.long	.L105-.L102
	.long	.L104-.L102
	.long	.L113-.L102
	.long	.L101-.L102
	.text
.L101:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L107
.L105:
	cmpl	$-1, -28(%rbp)
	jne	.L108
	movq	$4, -8(%rbp)
	jmp	.L107
.L108:
	movq	$2, -8(%rbp)
	jmp	.L107
.L104:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L107
.L112:
	nop
.L107:
	jmp	.L111
.L113:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	isempty, .-isempty
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
