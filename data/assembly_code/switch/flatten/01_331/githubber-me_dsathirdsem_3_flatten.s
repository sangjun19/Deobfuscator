	.file	"githubber-me_dsathirdsem_3_flatten.c"
	.text
	.globl	_TIG_IZ_fTK8_argv
	.bss
	.align 8
	.type	_TIG_IZ_fTK8_argv, @object
	.size	_TIG_IZ_fTK8_argv, 8
_TIG_IZ_fTK8_argv:
	.zero	8
	.globl	s
	.align 16
	.type	s, @object
	.size	s, 20
s:
	.zero	20
	.globl	_TIG_IZ_fTK8_envp
	.align 8
	.type	_TIG_IZ_fTK8_envp, @object
	.size	_TIG_IZ_fTK8_envp, 8
_TIG_IZ_fTK8_envp:
	.zero	8
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.globl	_TIG_IZ_fTK8_argc
	.align 4
	.type	_TIG_IZ_fTK8_argc, @object
	.size	_TIG_IZ_fTK8_argc, 4
_TIG_IZ_fTK8_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"\n~~~~Stack underflow~~~~"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L13:
	cmpq	$5, -8(%rbp)
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
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L14-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L9
	movq	$2, -8(%rbp)
	jmp	.L11
.L9:
	movq	$0, -8(%rbp)
	jmp	.L11
.L7:
	movl	$-1, %eax
	jmp	.L12
.L3:
	movl	-12(%rbp), %eax
	jmp	.L12
.L8:
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$5, -8(%rbp)
	jmp	.L11
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L11
.L14:
	nop
.L11:
	jmp	.L13
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	pop, .-pop
	.section	.rodata
.LC1:
	.string	"\nStack elements are:\n "
.LC2:
	.string	"\n~~~~Stack is empty~~~~"
.LC3:
	.string	"| %d |\n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$6, -8(%rbp)
.L31:
	cmpq	$10, -8(%rbp)
	ja	.L32
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L18(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L18(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L18:
	.long	.L33-.L18
	.long	.L23-.L18
	.long	.L32-.L18
	.long	.L33-.L18
	.long	.L32-.L18
	.long	.L21-.L18
	.long	.L20-.L18
	.long	.L32-.L18
	.long	.L19-.L18
	.long	.L32-.L18
	.long	.L17-.L18
	.text
.L19:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L25
.L23:
	cmpl	$0, -12(%rbp)
	js	.L26
	movq	$10, -8(%rbp)
	jmp	.L25
.L26:
	movq	$0, -8(%rbp)
	jmp	.L25
.L20:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L29
	movq	$5, -8(%rbp)
	jmp	.L25
.L29:
	movq	$8, -8(%rbp)
	jmp	.L25
.L21:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L25
.L17:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L25
.L32:
	nop
.L25:
	jmp	.L31
.L33:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	display, .-display
	.section	.rodata
.LC4:
	.string	"\n\n\n\n~~~~~~Menu~~~~~~ : "
	.align 8
.LC5:
	.string	"\n=>1.Push an Element to Stack and Overflow demo "
	.align 8
.LC6:
	.string	"\n=>2.Pop an Element from Stack and Underflow demo"
.LC7:
	.string	"\n=>3.Palindrome demo "
.LC8:
	.string	"\n=>4.Display "
.LC9:
	.string	"\n=>5.Exit"
.LC10:
	.string	"\nEnter your choice: "
.LC11:
	.string	"%d"
.LC12:
	.string	"\nElement popped is: %d"
	.align 8
.LC13:
	.string	"\nEnter an element to be pushed: "
.LC14:
	.string	"\nPlease enter valid choice "
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
.L35:
	movl	$0, -20(%rbp)
	jmp	.L36
.L37:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L36:
	cmpl	$4, -20(%rbp)
	jle	.L37
	nop
.L38:
	movq	$0, _TIG_IZ_fTK8_envp(%rip)
	nop
.L39:
	movq	$0, _TIG_IZ_fTK8_argv(%rip)
	nop
.L40:
	movl	$0, _TIG_IZ_fTK8_argc(%rip)
	nop
	nop
.L41:
.L42:
#APP
# 144 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fTK8--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_fTK8_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_fTK8_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_fTK8_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L67:
	cmpq	$17, -16(%rbp)
	ja	.L69
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L55-.L45
	.long	.L54-.L45
	.long	.L53-.L45
	.long	.L69-.L45
	.long	.L69-.L45
	.long	.L69-.L45
	.long	.L69-.L45
	.long	.L69-.L45
	.long	.L52-.L45
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L69-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L47-.L45
	.long	.L69-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L47:
	call	palindrome
	movq	$1, -16(%rbp)
	jmp	.L56
.L49:
	movl	$1, %edi
	call	exit@PLT
.L52:
	movl	-24(%rbp), %eax
	cmpl	$-1, %eax
	je	.L57
	movq	$9, -16(%rbp)
	jmp	.L56
.L57:
	movq	$1, -16(%rbp)
	jmp	.L56
.L54:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
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
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$13, -16(%rbp)
	jmp	.L56
.L46:
	call	display
	movq	$1, -16(%rbp)
	jmp	.L56
.L51:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L56
.L48:
	movl	-28(%rbp), %eax
	cmpl	$5, %eax
	ja	.L59
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L61(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L61(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L61:
	.long	.L59-.L61
	.long	.L65-.L61
	.long	.L64-.L61
	.long	.L63-.L61
	.long	.L62-.L61
	.long	.L60-.L61
	.text
.L60:
	movq	$12, -16(%rbp)
	jmp	.L66
.L62:
	movq	$16, -16(%rbp)
	jmp	.L66
.L63:
	movq	$14, -16(%rbp)
	jmp	.L66
.L64:
	movq	$10, -16(%rbp)
	jmp	.L66
.L65:
	movq	$17, -16(%rbp)
	jmp	.L66
.L59:
	movq	$2, -16(%rbp)
	nop
.L66:
	jmp	.L56
.L44:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$1, -16(%rbp)
	jmp	.L56
.L50:
	call	pop
	movl	%eax, -24(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L56
.L55:
	movq	$1, -16(%rbp)
	jmp	.L56
.L53:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L56
.L69:
	nop
.L56:
	jmp	.L67
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
.LC15:
	.string	"\n~~~~Stack overflow~~~~"
	.text
	.globl	push
	.type	push, @function
push:
.LFB5:
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
.L82:
	cmpq	$4, -8(%rbp)
	ja	.L83
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L73(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L73(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L73:
	.long	.L77-.L73
	.long	.L84-.L73
	.long	.L84-.L73
	.long	.L74-.L73
	.long	.L72-.L73
	.text
.L72:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L78
.L74:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	s(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$2, -8(%rbp)
	jmp	.L78
.L77:
	movl	top(%rip), %eax
	cmpl	$4, %eax
	jne	.L80
	movq	$4, -8(%rbp)
	jmp	.L78
.L80:
	movq	$3, -8(%rbp)
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
.LFE5:
	.size	push, .-push
	.section	.rodata
.LC16:
	.string	"\nIt is palindrome number"
	.align 8
.LC17:
	.string	"\nReverse of stack content are:"
.LC18:
	.string	"\nStack content are:"
	.align 8
.LC19:
	.string	"\nIt is not a palindrome number"
	.text
	.globl	palindrome
	.type	palindrome, @function
palindrome:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$13, -8(%rbp)
.L116:
	cmpq	$25, -8(%rbp)
	ja	.L117
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L88(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L88(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L88:
	.long	.L103-.L88
	.long	.L117-.L88
	.long	.L102-.L88
	.long	.L101-.L88
	.long	.L117-.L88
	.long	.L117-.L88
	.long	.L117-.L88
	.long	.L117-.L88
	.long	.L100-.L88
	.long	.L99-.L88
	.long	.L98-.L88
	.long	.L97-.L88
	.long	.L96-.L88
	.long	.L95-.L88
	.long	.L117-.L88
	.long	.L117-.L88
	.long	.L118-.L88
	.long	.L93-.L88
	.long	.L92-.L88
	.long	.L117-.L88
	.long	.L117-.L88
	.long	.L91-.L88
	.long	.L117-.L88
	.long	.L90-.L88
	.long	.L89-.L88
	.long	.L87-.L88
	.text
.L92:
	movl	$0, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L104
.L87:
	cmpl	$1, -16(%rbp)
	jne	.L105
	movq	$8, -8(%rbp)
	jmp	.L104
.L105:
	movq	$0, -8(%rbp)
	jmp	.L104
.L96:
	addl	$1, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L104
.L100:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L104
.L90:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -12(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L104
.L101:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	top(%rip), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	s(%rip), %rax
	movl	(%rcx,%rax), %eax
	cmpl	%eax, %edx
	je	.L107
	movq	$10, -8(%rbp)
	jmp	.L104
.L107:
	movq	$12, -8(%rbp)
	jmp	.L104
.L89:
	movl	$1, -16(%rbp)
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L104
.L91:
	movl	top(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L110
	movq	$2, -8(%rbp)
	jmp	.L104
.L110:
	movq	$18, -8(%rbp)
	jmp	.L104
.L97:
	cmpl	$0, -12(%rbp)
	js	.L112
	movq	$17, -8(%rbp)
	jmp	.L104
.L112:
	movq	$23, -8(%rbp)
	jmp	.L104
.L99:
	movl	top(%rip), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -12(%rbp)
	jg	.L114
	movq	$3, -8(%rbp)
	jmp	.L104
.L114:
	movq	$25, -8(%rbp)
	jmp	.L104
.L95:
	movq	$24, -8(%rbp)
	jmp	.L104
.L93:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L104
.L98:
	movl	$0, -16(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L104
.L103:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L104
.L102:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L104
.L117:
	nop
.L104:
	jmp	.L116
.L118:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	palindrome, .-palindrome
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
