	.file	"halakundi_DSA-Program_3_flatten.c"
	.text
	.globl	stack
	.bss
	.align 16
	.type	stack, @object
	.size	stack, 20
stack:
	.zero	20
	.globl	ele
	.align 4
	.type	ele, @object
	.size	ele, 4
ele:
	.zero	4
	.globl	_TIG_IZ_R95T_argc
	.align 4
	.type	_TIG_IZ_R95T_argc, @object
	.size	_TIG_IZ_R95T_argc, 4
_TIG_IZ_R95T_argc:
	.zero	4
	.globl	_TIG_IZ_R95T_argv
	.align 8
	.type	_TIG_IZ_R95T_argv, @object
	.size	_TIG_IZ_R95T_argv, 8
_TIG_IZ_R95T_argv:
	.zero	8
	.globl	_TIG_IZ_R95T_envp
	.align 8
	.type	_TIG_IZ_R95T_envp, @object
	.size	_TIG_IZ_R95T_envp, 8
_TIG_IZ_R95T_envp:
	.zero	8
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.section	.rodata
.LC0:
	.string	"popped element: %d\n"
.LC1:
	.string	"enter element to push: "
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"Menu \n 1.push \n 2.pop \n 3.status \n 4.display \n 5.exit \n enter your choice: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
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
	movl	$-1, top(%rip)
	nop
.L2:
	movl	$0, ele(%rip)
	nop
.L3:
	movl	$0, -20(%rbp)
	jmp	.L4
.L5:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L4:
	cmpl	$4, -20(%rbp)
	jle	.L5
	nop
.L6:
	movq	$0, _TIG_IZ_R95T_envp(%rip)
	nop
.L7:
	movq	$0, _TIG_IZ_R95T_argv(%rip)
	nop
.L8:
	movl	$0, _TIG_IZ_R95T_argc(%rip)
	nop
	nop
.L9:
.L10:
#APP
# 131 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-R95T--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_R95T_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_R95T_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_R95T_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L31:
	cmpq	$16, -16(%rbp)
	ja	.L33
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L13(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L13(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L13:
	.long	.L21-.L13
	.long	.L33-.L13
	.long	.L20-.L13
	.long	.L33-.L13
	.long	.L19-.L13
	.long	.L33-.L13
	.long	.L18-.L13
	.long	.L17-.L13
	.long	.L33-.L13
	.long	.L16-.L13
	.long	.L15-.L13
	.long	.L33-.L13
	.long	.L14-.L13
	.long	.L33-.L13
	.long	.L33-.L13
	.long	.L33-.L13
	.long	.L12-.L13
	.text
.L19:
	movl	$0, %edi
	call	exit@PLT
.L14:
	movl	-24(%rbp), %eax
	cmpl	$5, %eax
	ja	.L22
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L22-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L23:
	movq	$4, -16(%rbp)
	jmp	.L29
.L25:
	movq	$9, -16(%rbp)
	jmp	.L29
.L26:
	movq	$0, -16(%rbp)
	jmp	.L29
.L27:
	movq	$16, -16(%rbp)
	jmp	.L29
.L28:
	movq	$6, -16(%rbp)
	jmp	.L29
.L22:
	movq	$2, -16(%rbp)
	nop
.L29:
	jmp	.L30
.L12:
	call	pop
	movl	%eax, ele(%rip)
	movl	ele(%rip), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L30
.L16:
	call	display
	movq	$10, -16(%rbp)
	jmp	.L30
.L18:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	ele(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	ele(%rip), %eax
	movl	%eax, %edi
	call	push
	movq	$10, -16(%rbp)
	jmp	.L30
.L15:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L21:
	call	status
	movq	$10, -16(%rbp)
	jmp	.L30
.L17:
	movq	$10, -16(%rbp)
	jmp	.L30
.L20:
	movq	$10, -16(%rbp)
	jmp	.L30
.L33:
	nop
.L30:
	jmp	.L31
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC4:
	.string	"Stack is full"
.LC5:
	.string	"Stack is empty"
	.text
	.globl	status
	.type	status, @function
status:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L49:
	cmpq	$5, -8(%rbp)
	ja	.L50
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L37(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L37(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L37:
	.long	.L42-.L37
	.long	.L41-.L37
	.long	.L51-.L37
	.long	.L39-.L37
	.long	.L38-.L37
	.long	.L36-.L37
	.text
.L38:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L43
.L41:
	call	display
	movq	$2, -8(%rbp)
	jmp	.L43
.L39:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L44
	movq	$5, -8(%rbp)
	jmp	.L43
.L44:
	movq	$0, -8(%rbp)
	jmp	.L43
.L36:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L43
.L42:
	movl	top(%rip), %eax
	cmpl	$4, %eax
	jne	.L46
	movq	$4, -8(%rbp)
	jmp	.L43
.L46:
	movq	$1, -8(%rbp)
	jmp	.L43
.L50:
	nop
.L43:
	jmp	.L49
.L51:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	status, .-status
	.globl	pop
	.type	pop, @function
pop:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L64:
	cmpq	$5, -8(%rbp)
	ja	.L65
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L55(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L55(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L55:
	.long	.L59-.L55
	.long	.L58-.L55
	.long	.L57-.L55
	.long	.L65-.L55
	.long	.L56-.L55
	.long	.L54-.L55
	.text
.L56:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L60
	movq	$1, -8(%rbp)
	jmp	.L62
.L60:
	movq	$0, -8(%rbp)
	jmp	.L62
.L58:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L62
.L54:
	movl	$0, %eax
	jmp	.L63
.L59:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$2, -8(%rbp)
	jmp	.L62
.L57:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L63
.L65:
	nop
.L62:
	jmp	.L64
.L63:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	pop, .-pop
	.section	.rodata
.LC6:
	.string	"%d\n"
.LC7:
	.string	"Stack elements are:"
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
	movq	$3, -8(%rbp)
.L78:
	cmpq	$6, -8(%rbp)
	ja	.L79
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L69(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L69(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L69:
	.long	.L73-.L69
	.long	.L79-.L69
	.long	.L79-.L69
	.long	.L72-.L69
	.long	.L71-.L69
	.long	.L70-.L69
	.long	.L80-.L69
	.text
.L71:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L74
.L72:
	movq	$0, -8(%rbp)
	jmp	.L74
.L70:
	cmpl	$0, -12(%rbp)
	js	.L76
	movq	$4, -8(%rbp)
	jmp	.L74
.L76:
	movq	$6, -8(%rbp)
	jmp	.L74
.L73:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L74
.L79:
	nop
.L74:
	jmp	.L78
.L80:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	display, .-display
	.globl	push
	.type	push, @function
push:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L91:
	cmpq	$4, -8(%rbp)
	je	.L92
	cmpq	$4, -8(%rbp)
	ja	.L93
	cmpq	$3, -8(%rbp)
	je	.L84
	cmpq	$3, -8(%rbp)
	ja	.L93
	cmpq	$0, -8(%rbp)
	je	.L85
	cmpq	$1, -8(%rbp)
	je	.L86
	jmp	.L93
.L86:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L88
.L84:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	stack(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$4, -8(%rbp)
	jmp	.L88
.L85:
	movl	top(%rip), %eax
	cmpl	$4, %eax
	jne	.L89
	movq	$1, -8(%rbp)
	jmp	.L88
.L89:
	movq	$3, -8(%rbp)
	jmp	.L88
.L93:
	nop
.L88:
	jmp	.L91
.L92:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	push, .-push
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
