	.file	"vaibhavkrishna-bhosle_DSA_3_flatten.c"
	.text
	.globl	_TIG_IZ_MwM7_envp
	.bss
	.align 8
	.type	_TIG_IZ_MwM7_envp, @object
	.size	_TIG_IZ_MwM7_envp, 8
_TIG_IZ_MwM7_envp:
	.zero	8
	.globl	stack
	.align 16
	.type	stack, @object
	.size	stack, 20
stack:
	.zero	20
	.globl	_TIG_IZ_MwM7_argv
	.align 8
	.type	_TIG_IZ_MwM7_argv, @object
	.size	_TIG_IZ_MwM7_argv, 8
_TIG_IZ_MwM7_argv:
	.zero	8
	.globl	_TIG_IZ_MwM7_argc
	.align 4
	.type	_TIG_IZ_MwM7_argc, @object
	.size	_TIG_IZ_MwM7_argc, 4
_TIG_IZ_MwM7_argc:
	.zero	4
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"----------------------------------Deleted element is %d\n"
	.align 8
.LC1:
	.string	"----------------------------------Stack is empty"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L11:
	cmpq	$4, -8(%rbp)
	je	.L2
	cmpq	$4, -8(%rbp)
	ja	.L12
	cmpq	$3, -8(%rbp)
	je	.L13
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	je	.L6
	jmp	.L12
.L2:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L7
	movq	$0, -8(%rbp)
	jmp	.L9
.L7:
	movq	$1, -8(%rbp)
	jmp	.L9
.L6:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L9
.L5:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L9
.L12:
	nop
.L9:
	jmp	.L11
.L13:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	pop, .-pop
	.section	.rodata
	.align 8
.LC2:
	.string	"----------------------------------Stack is full"
.LC3:
	.string	"Enter data: "
.LC4:
	.string	"%d"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -16(%rbp)
.L24:
	cmpq	$4, -16(%rbp)
	je	.L15
	cmpq	$4, -16(%rbp)
	ja	.L27
	cmpq	$3, -16(%rbp)
	je	.L28
	cmpq	$3, -16(%rbp)
	ja	.L27
	cmpq	$0, -16(%rbp)
	je	.L18
	cmpq	$1, -16(%rbp)
	je	.L19
	jmp	.L27
.L15:
	movl	top(%rip), %eax
	cmpl	$4, %eax
	jne	.L20
	movq	$1, -16(%rbp)
	jmp	.L22
.L20:
	movq	$0, -16(%rbp)
	jmp	.L22
.L19:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L22
.L18:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %edx
	movl	-20(%rbp), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	stack(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$3, -16(%rbp)
	jmp	.L22
.L27:
	nop
.L22:
	jmp	.L24
.L28:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L26
	call	__stack_chk_fail@PLT
.L26:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	push, .-push
	.section	.rodata
	.align 8
.LC5:
	.string	"----------------------------------Palindrome"
	.align 8
.LC6:
	.string	"----------------------------------Not palindrome"
	.text
	.globl	palindrome
	.type	palindrome, @function
palindrome:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L47:
	cmpq	$10, -8(%rbp)
	ja	.L48
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L40-.L32
	.long	.L48-.L32
	.long	.L39-.L32
	.long	.L38-.L32
	.long	.L48-.L32
	.long	.L37-.L32
	.long	.L36-.L32
	.long	.L49-.L32
	.long	.L34-.L32
	.long	.L49-.L32
	.long	.L31-.L32
	.text
.L34:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	stack(%rip), %rax
	movl	(%rcx,%rax), %eax
	cmpl	%eax, %edx
	je	.L41
	movq	$6, -8(%rbp)
	jmp	.L43
.L41:
	movq	$0, -8(%rbp)
	jmp	.L43
.L38:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L43
.L36:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -8(%rbp)
	jmp	.L43
.L37:
	movl	-16(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jge	.L45
	movq	$8, -8(%rbp)
	jmp	.L43
.L45:
	movq	$3, -8(%rbp)
	jmp	.L43
.L31:
	movl	$0, -16(%rbp)
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L43
.L40:
	addl	$1, -16(%rbp)
	subl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L43
.L39:
	movq	$10, -8(%rbp)
	jmp	.L43
.L48:
	nop
.L43:
	jmp	.L47
.L49:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	palindrome, .-palindrome
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
	movq	$8, -8(%rbp)
.L65:
	cmpq	$8, -8(%rbp)
	ja	.L66
	movq	-8(%rbp), %rax
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
	.long	.L66-.L53
	.long	.L58-.L53
	.long	.L57-.L53
	.long	.L66-.L53
	.long	.L67-.L53
	.long	.L55-.L53
	.long	.L54-.L53
	.long	.L66-.L53
	.long	.L52-.L53
	.text
.L52:
	movl	top(%rip), %eax
	cmpl	$-1, %eax
	jne	.L60
	movq	$5, -8(%rbp)
	jmp	.L62
.L60:
	movq	$6, -8(%rbp)
	jmp	.L62
.L58:
	cmpl	$0, -12(%rbp)
	js	.L63
	movq	$2, -8(%rbp)
	jmp	.L62
.L63:
	movq	$4, -8(%rbp)
	jmp	.L62
.L54:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L62
.L55:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L62
.L57:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L62
.L66:
	nop
.L62:
	jmp	.L65
.L67:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	display, .-display
	.section	.rodata
.LC7:
	.string	"Invalid choice"
	.align 8
.LC8:
	.string	"\n----------------------------------\n1. Push\n2. Pop\n3. Display\n4. Palindrome\n5. Exit\nEnter your choice: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB10:
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
.L69:
	movl	$0, -20(%rbp)
	jmp	.L70
.L71:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L70:
	cmpl	$4, -20(%rbp)
	jle	.L71
	nop
.L72:
	movq	$0, _TIG_IZ_MwM7_envp(%rip)
	nop
.L73:
	movq	$0, _TIG_IZ_MwM7_argv(%rip)
	nop
.L74:
	movl	$0, _TIG_IZ_MwM7_argc(%rip)
	nop
	nop
.L75:
.L76:
#APP
# 99 "vaibhavkrishna-bhosle_DSA_3.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-MwM7--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_MwM7_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_MwM7_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_MwM7_envp(%rip)
	nop
	movq	$10, -16(%rbp)
.L97:
	cmpq	$11, -16(%rbp)
	ja	.L99
	movq	-16(%rbp), %rax
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
	.long	.L99-.L79
	.long	.L87-.L79
	.long	.L86-.L79
	.long	.L85-.L79
	.long	.L99-.L79
	.long	.L84-.L79
	.long	.L83-.L79
	.long	.L82-.L79
	.long	.L99-.L79
	.long	.L81-.L79
	.long	.L80-.L79
	.long	.L78-.L79
	.text
.L87:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L88
.L85:
	call	display
	movq	$6, -16(%rbp)
	jmp	.L88
.L78:
	movl	-24(%rbp), %eax
	cmpl	$5, %eax
	ja	.L89
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L91(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L91(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L91:
	.long	.L89-.L91
	.long	.L95-.L91
	.long	.L94-.L91
	.long	.L93-.L91
	.long	.L92-.L91
	.long	.L90-.L91
	.text
.L90:
	movq	$2, -16(%rbp)
	jmp	.L96
.L92:
	movq	$5, -16(%rbp)
	jmp	.L96
.L93:
	movq	$3, -16(%rbp)
	jmp	.L96
.L94:
	movq	$9, -16(%rbp)
	jmp	.L96
.L95:
	movq	$7, -16(%rbp)
	jmp	.L96
.L89:
	movq	$1, -16(%rbp)
	nop
.L96:
	jmp	.L88
.L81:
	call	pop
	movq	$6, -16(%rbp)
	jmp	.L88
.L83:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -16(%rbp)
	jmp	.L88
.L84:
	call	palindrome
	movq	$6, -16(%rbp)
	jmp	.L88
.L80:
	movq	$6, -16(%rbp)
	jmp	.L88
.L82:
	call	push
	movq	$6, -16(%rbp)
	jmp	.L88
.L86:
	movl	$0, %edi
	call	exit@PLT
.L99:
	nop
.L88:
	jmp	.L97
	.cfi_endproc
.LFE10:
	.size	main, .-main
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
