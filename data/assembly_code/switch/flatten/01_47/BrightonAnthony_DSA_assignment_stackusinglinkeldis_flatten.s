	.file	"BrightonAnthony_DSA_assignment_stackusinglinkeldis_flatten.c"
	.text
	.globl	_TIG_IZ_EOCt_envp
	.bss
	.align 8
	.type	_TIG_IZ_EOCt_envp, @object
	.size	_TIG_IZ_EOCt_envp, 8
_TIG_IZ_EOCt_envp:
	.zero	8
	.globl	_TIG_IZ_EOCt_argc
	.align 4
	.type	_TIG_IZ_EOCt_argc, @object
	.size	_TIG_IZ_EOCt_argc, 4
_TIG_IZ_EOCt_argc:
	.zero	4
	.globl	top
	.align 8
	.type	top, @object
	.size	top, 8
top:
	.zero	8
	.globl	_TIG_IZ_EOCt_argv
	.align 8
	.type	_TIG_IZ_EOCt_argv, @object
	.size	_TIG_IZ_EOCt_argv, 8
_TIG_IZ_EOCt_argv:
	.zero	8
	.text
	.globl	getnode
	.type	getnode, @function
getnode:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	movq	-16(%rbp), %rax
	jmp	.L7
.L2:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	getnode, .-getnode
	.section	.rodata
.LC0:
	.string	"\n List is EMTPTY"
	.align 8
.LC1:
	.string	"%d element deleted successfully"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$3, -16(%rbp)
.L18:
	cmpq	$3, -16(%rbp)
	je	.L10
	cmpq	$3, -16(%rbp)
	ja	.L19
	cmpq	$2, -16(%rbp)
	je	.L20
	cmpq	$2, -16(%rbp)
	ja	.L19
	cmpq	$0, -16(%rbp)
	je	.L13
	cmpq	$1, -16(%rbp)
	jne	.L19
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L14
.L10:
	movq	top(%rip), %rax
	testq	%rax, %rax
	jne	.L15
	movq	$1, -16(%rbp)
	jmp	.L14
.L15:
	movq	$0, -16(%rbp)
	jmp	.L14
.L13:
	movq	top(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	top(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	top(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, top(%rip)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	freenode
	movq	$2, -16(%rbp)
	jmp	.L14
.L19:
	nop
.L14:
	jmp	.L18
.L20:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	pop, .-pop
	.globl	push
	.type	push, @function
push:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$2, -16(%rbp)
.L27:
	cmpq	$2, -16(%rbp)
	je	.L22
	cmpq	$2, -16(%rbp)
	ja	.L29
	cmpq	$0, -16(%rbp)
	je	.L24
	cmpq	$1, -16(%rbp)
	jne	.L29
	jmp	.L28
.L24:
	call	getnode
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, (%rax)
	movq	top(%rip), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, top(%rip)
	movq	$1, -16(%rbp)
	jmp	.L26
.L22:
	movq	$0, -16(%rbp)
	jmp	.L26
.L29:
	nop
.L26:
	jmp	.L27
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	push, .-push
	.section	.rodata
.LC2:
	.string	"-> %d"
.LC3:
	.string	"\ntop"
	.text
	.globl	display
	.type	display, @function
display:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$5, -8(%rbp)
.L42:
	cmpq	$7, -8(%rbp)
	ja	.L43
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L33(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L33(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L33:
	.long	.L37-.L33
	.long	.L44-.L33
	.long	.L35-.L33
	.long	.L43-.L33
	.long	.L43-.L33
	.long	.L34-.L33
	.long	.L43-.L33
	.long	.L32-.L33
	.text
.L34:
	movq	$2, -8(%rbp)
	jmp	.L39
.L37:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L39
.L32:
	cmpq	$0, -16(%rbp)
	je	.L40
	movq	$0, -8(%rbp)
	jmp	.L39
.L40:
	movq	$1, -8(%rbp)
	jmp	.L39
.L35:
	movq	top(%rip), %rax
	movq	%rax, -16(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L39
.L43:
	nop
.L39:
	jmp	.L42
.L44:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	display, .-display
	.section	.rodata
	.align 8
.LC4:
	.string	"\n Enter the element to be pused into the stack: "
.LC5:
	.string	"%d"
	.align 8
.LC6:
	.string	"\nStack Operation using Linked List"
	.align 8
.LC7:
	.string	"\n1. Push an element into the stack."
	.align 8
.LC8:
	.string	"\n2. Pop out an element from the stack."
	.align 8
.LC9:
	.string	"\n3. Display an element from the stack."
.LC10:
	.string	"\n4. Exit."
.LC11:
	.string	"\n Enter your Choice:"
.LC12:
	.string	"Wrong Choice"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, top(%rip)
	nop
.L46:
	movq	$0, _TIG_IZ_EOCt_envp(%rip)
	nop
.L47:
	movq	$0, _TIG_IZ_EOCt_argv(%rip)
	nop
.L48:
	movl	$0, _TIG_IZ_EOCt_argc(%rip)
	nop
	nop
.L49:
.L50:
#APP
# 137 "BrightonAnthony_DSA_assignment_stackusinglinkeldis.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EOCt--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_EOCt_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_EOCt_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_EOCt_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L68:
	cmpq	$15, -16(%rbp)
	ja	.L70
	movq	-16(%rbp), %rax
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
	.long	.L70-.L53
	.long	.L60-.L53
	.long	.L59-.L53
	.long	.L70-.L53
	.long	.L58-.L53
	.long	.L70-.L53
	.long	.L57-.L53
	.long	.L56-.L53
	.long	.L70-.L53
	.long	.L55-.L53
	.long	.L70-.L53
	.long	.L70-.L53
	.long	.L70-.L53
	.long	.L54-.L53
	.long	.L70-.L53
	.long	.L52-.L53
	.text
.L58:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$1, -16(%rbp)
	jmp	.L61
.L52:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L62
	cmpl	$4, %eax
	jg	.L63
	cmpl	$3, %eax
	je	.L64
	cmpl	$3, %eax
	jg	.L63
	cmpl	$1, %eax
	je	.L65
	cmpl	$2, %eax
	je	.L66
	jmp	.L63
.L62:
	movq	$13, -16(%rbp)
	jmp	.L67
.L64:
	movq	$7, -16(%rbp)
	jmp	.L67
.L66:
	movq	$9, -16(%rbp)
	jmp	.L67
.L65:
	movq	$4, -16(%rbp)
	jmp	.L67
.L63:
	movq	$2, -16(%rbp)
	nop
.L67:
	jmp	.L61
.L60:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$15, -16(%rbp)
	jmp	.L61
.L55:
	call	pop
	movq	$1, -16(%rbp)
	jmp	.L61
.L54:
	movl	$1, %edi
	call	exit@PLT
.L57:
	movq	$1, -16(%rbp)
	jmp	.L61
.L56:
	call	display
	movq	$1, -16(%rbp)
	jmp	.L61
.L59:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L61
.L70:
	nop
.L61:
	jmp	.L68
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.globl	freenode
	.type	freenode, @function
freenode:
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
	movq	$0, -8(%rbp)
.L76:
	cmpq	$0, -8(%rbp)
	je	.L72
	cmpq	$1, -8(%rbp)
	jne	.L78
	jmp	.L77
.L72:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -8(%rbp)
	jmp	.L75
.L78:
	nop
.L75:
	jmp	.L76
.L77:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	freenode, .-freenode
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
