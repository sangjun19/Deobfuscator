	.file	"Vaibhav22p-qw_DSA-Practice_stack_flatten.c"
	.text
	.globl	_TIG_IZ_Zpbn_envp
	.bss
	.align 8
	.type	_TIG_IZ_Zpbn_envp, @object
	.size	_TIG_IZ_Zpbn_envp, 8
_TIG_IZ_Zpbn_envp:
	.zero	8
	.globl	_TIG_IZ_Zpbn_argv
	.align 8
	.type	_TIG_IZ_Zpbn_argv, @object
	.size	_TIG_IZ_Zpbn_argv, 8
_TIG_IZ_Zpbn_argv:
	.zero	8
	.globl	_TIG_IZ_Zpbn_argc
	.align 4
	.type	_TIG_IZ_Zpbn_argc, @object
	.size	_TIG_IZ_Zpbn_argc, 4
_TIG_IZ_Zpbn_argc:
	.zero	4
	.globl	stack
	.align 16
	.type	stack, @object
	.size	stack, 20
stack:
	.zero	20
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Oops! Stack is empty, unable to POP."
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L10:
	cmpq	$3, -8(%rbp)
	je	.L2
	cmpq	$3, -8(%rbp)
	ja	.L11
	cmpq	$2, -8(%rbp)
	je	.L12
	cmpq	$2, -8(%rbp)
	ja	.L11
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	jne	.L11
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L6
	movq	$3, -8(%rbp)
	jmp	.L8
.L6:
	movq	$0, -8(%rbp)
	jmp	.L8
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L8
.L5:
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$2, -8(%rbp)
	jmp	.L8
.L11:
	nop
.L8:
	jmp	.L10
.L12:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	pop, .-pop
	.section	.rodata
.LC1:
	.string	"1.PUSH"
.LC2:
	.string	"2. POP"
.LC3:
	.string	"3. Exit"
	.text
	.globl	menu
	.type	menu, @function
menu:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L19:
	cmpq	$2, -8(%rbp)
	je	.L14
	cmpq	$2, -8(%rbp)
	ja	.L20
	cmpq	$0, -8(%rbp)
	je	.L21
	cmpq	$1, -8(%rbp)
	jne	.L20
	movq	$2, -8(%rbp)
	jmp	.L17
.L14:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L17
.L20:
	nop
.L17:
	jmp	.L19
.L21:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	menu, .-menu
	.section	.rodata
.LC4:
	.string	"\n| Final List |"
.LC5:
	.string	">>> "
.LC6:
	.string	"%d"
.LC7:
	.string	"Enter Value : "
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
.L23:
	movl	$0, -20(%rbp)
	jmp	.L24
.L25:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L24:
	cmpl	$4, -20(%rbp)
	jle	.L25
	nop
.L26:
	movq	$0, _TIG_IZ_Zpbn_envp(%rip)
	nop
.L27:
	movq	$0, _TIG_IZ_Zpbn_argv(%rip)
	nop
.L28:
	movl	$0, _TIG_IZ_Zpbn_argc(%rip)
	nop
	nop
.L29:
.L30:
#APP
# 104 "Vaibhav22p-qw_DSA-Practice_stack.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Zpbn--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Zpbn_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Zpbn_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Zpbn_envp(%rip)
	nop
	movq	$8, -16(%rbp)
.L49:
	cmpq	$16, -16(%rbp)
	ja	.L52
	movq	-16(%rbp), %rax
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
	.long	.L52-.L33
	.long	.L52-.L33
	.long	.L41-.L33
	.long	.L40-.L33
	.long	.L39-.L33
	.long	.L52-.L33
	.long	.L38-.L33
	.long	.L37-.L33
	.long	.L36-.L33
	.long	.L52-.L33
	.long	.L52-.L33
	.long	.L52-.L33
	.long	.L52-.L33
	.long	.L35-.L33
	.long	.L34-.L33
	.long	.L52-.L33
	.long	.L32-.L33
	.text
.L39:
	movl	-28(%rbp), %eax
	cmpl	$3, %eax
	je	.L42
	movq	$13, -16(%rbp)
	jmp	.L44
.L42:
	movq	$3, -16(%rbp)
	jmp	.L44
.L34:
	movl	-28(%rbp), %eax
	cmpl	$1, %eax
	je	.L45
	cmpl	$2, %eax
	jne	.L46
	movq	$16, -16(%rbp)
	jmp	.L47
.L45:
	movq	$6, -16(%rbp)
	jmp	.L47
.L46:
	movq	$7, -16(%rbp)
	nop
.L47:
	jmp	.L44
.L36:
	call	menu
	movq	$13, -16(%rbp)
	jmp	.L44
.L40:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	print
	movq	$2, -16(%rbp)
	jmp	.L44
.L32:
	call	pop
	call	print
	movq	$4, -16(%rbp)
	jmp	.L44
.L35:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -16(%rbp)
	jmp	.L44
.L38:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	push
	call	print
	movq	$4, -16(%rbp)
	jmp	.L44
.L37:
	movq	$4, -16(%rbp)
	jmp	.L44
.L41:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L50
	jmp	.L51
.L52:
	nop
.L44:
	jmp	.L49
.L51:
	call	__stack_chk_fail@PLT
.L50:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
.LC8:
	.string	"Oops! Unable to PUSH."
	.text
	.globl	push
	.type	push, @function
push:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$4, -8(%rbp)
.L63:
	cmpq	$4, -8(%rbp)
	je	.L54
	cmpq	$4, -8(%rbp)
	ja	.L64
	cmpq	$3, -8(%rbp)
	je	.L56
	cmpq	$3, -8(%rbp)
	ja	.L64
	cmpq	$1, -8(%rbp)
	je	.L65
	cmpq	$2, -8(%rbp)
	je	.L58
	jmp	.L64
.L54:
	movl	top(%rip), %eax
	cmpl	$4, %eax
	jne	.L59
	movq	$2, -8(%rbp)
	jmp	.L61
.L59:
	movq	$3, -8(%rbp)
	jmp	.L61
.L56:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	stack(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$1, -8(%rbp)
	jmp	.L61
.L58:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L61
.L64:
	nop
.L61:
	jmp	.L63
.L65:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	push, .-push
	.section	.rodata
.LC9:
	.string	"%d\n"
.LC10:
	.string	"Empty"
	.text
	.globl	print
	.type	print, @function
print:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L82:
	cmpq	$9, -8(%rbp)
	ja	.L83
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
	.long	.L83-.L69
	.long	.L84-.L69
	.long	.L74-.L69
	.long	.L83-.L69
	.long	.L73-.L69
	.long	.L72-.L69
	.long	.L71-.L69
	.long	.L83-.L69
	.long	.L84-.L69
	.long	.L68-.L69
	.text
.L73:
	cmpl	$0, -12(%rbp)
	js	.L76
	movq	$6, -8(%rbp)
	jmp	.L78
.L76:
	movq	$8, -8(%rbp)
	jmp	.L78
.L68:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L78
.L71:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L78
.L72:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L78
.L74:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L80
	movq	$5, -8(%rbp)
	jmp	.L78
.L80:
	movq	$9, -8(%rbp)
	jmp	.L78
.L83:
	nop
.L78:
	jmp	.L82
.L84:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	print, .-print
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
