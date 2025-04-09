	.file	"Pranav016_Hacktoberfest-PR-for-beginners_input_in_beginning_flatten.c"
	.text
	.globl	_TIG_IZ_4JuA_envp
	.bss
	.align 8
	.type	_TIG_IZ_4JuA_envp, @object
	.size	_TIG_IZ_4JuA_envp, 8
_TIG_IZ_4JuA_envp:
	.zero	8
	.globl	head
	.align 8
	.type	head, @object
	.size	head, 8
head:
	.zero	8
	.globl	newnode
	.align 8
	.type	newnode, @object
	.size	newnode, 8
newnode:
	.zero	8
	.globl	_TIG_IZ_4JuA_argc
	.align 4
	.type	_TIG_IZ_4JuA_argc, @object
	.size	_TIG_IZ_4JuA_argc, 4
_TIG_IZ_4JuA_argc:
	.zero	4
	.globl	_TIG_IZ_4JuA_argv
	.align 8
	.type	_TIG_IZ_4JuA_argv, @object
	.size	_TIG_IZ_4JuA_argv, 8
_TIG_IZ_4JuA_argv:
	.zero	8
	.globl	temp
	.align 8
	.type	temp, @object
	.size	temp, 8
temp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d -> "
.LC1:
	.string	"NULL"
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
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$0, -8(%rbp)
.L14:
	cmpq	$8, -8(%rbp)
	ja	.L15
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
	.long	.L9-.L4
	.long	.L16-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L15-.L4
	.long	.L15-.L4
	.long	.L15-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L10
	movq	$7, -8(%rbp)
	jmp	.L12
.L10:
	movq	$2, -8(%rbp)
	jmp	.L12
.L6:
	movq	-24(%rbp), %rax
	movq	%rax, temp(%rip)
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L12
.L9:
	movq	$3, -8(%rbp)
	jmp	.L12
.L5:
	movq	temp(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	temp(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, temp(%rip)
	addl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L12
.L7:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L12
.L15:
	nop
.L12:
	jmp	.L14
.L16:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	display, .-display
	.section	.rodata
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"enter the elements of the node"
	.text
	.globl	input_node
	.type	input_node, @function
input_node:
.LFB3:
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
	movq	$7, -16(%rbp)
.L29:
	cmpq	$7, -16(%rbp)
	ja	.L30
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L20(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L20(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L20:
	.long	.L24-.L20
	.long	.L23-.L20
	.long	.L22-.L20
	.long	.L30-.L20
	.long	.L31-.L20
	.long	.L30-.L20
	.long	.L30-.L20
	.long	.L19-.L20
	.text
.L23:
	movl	$8, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, newnode(%rip)
	movq	newnode(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	newnode(%rip), %rax
	movq	$0, 8(%rax)
	movq	temp(%rip), %rax
	movq	newnode(%rip), %rdx
	movq	%rdx, 8(%rax)
	movq	temp(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, temp(%rip)
	addl	$1, -20(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L26
.L24:
	movl	-20(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L27
	movq	$1, -16(%rbp)
	jmp	.L26
.L27:
	movq	$4, -16(%rbp)
	jmp	.L26
.L19:
	movq	$2, -16(%rbp)
	jmp	.L26
.L22:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-40(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-40(%rbp), %rax
	movq	%rax, temp(%rip)
	movl	$1, -20(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L26
.L30:
	nop
.L26:
	jmp	.L29
.L31:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	input_node, .-input_node
	.section	.rodata
.LC4:
	.string	"enter the size of the node"
	.align 8
.LC5:
	.string	"choose between the two options\n 1. for creation of node\n 2. for display of node\n 3. for inserting at beginning"
.LC6:
	.string	"1 to contiue and 0 to exit"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
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
	movq	$0, newnode(%rip)
	nop
.L33:
	movq	$0, temp(%rip)
	nop
.L34:
	movq	$0, head(%rip)
	nop
.L35:
	movq	$0, _TIG_IZ_4JuA_envp(%rip)
	nop
.L36:
	movq	$0, _TIG_IZ_4JuA_argv(%rip)
	nop
.L37:
	movl	$0, _TIG_IZ_4JuA_argc(%rip)
	nop
	nop
.L38:
.L39:
#APP
# 149 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4JuA--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_4JuA_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_4JuA_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_4JuA_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L62:
	cmpq	$18, -24(%rbp)
	ja	.L65
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L42(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L42(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L42:
	.long	.L52-.L42
	.long	.L51-.L42
	.long	.L50-.L42
	.long	.L65-.L42
	.long	.L65-.L42
	.long	.L65-.L42
	.long	.L65-.L42
	.long	.L65-.L42
	.long	.L49-.L42
	.long	.L65-.L42
	.long	.L48-.L42
	.long	.L65-.L42
	.long	.L47-.L42
	.long	.L65-.L42
	.long	.L46-.L42
	.long	.L45-.L42
	.long	.L44-.L42
	.long	.L43-.L42
	.long	.L41-.L42
	.text
.L41:
	movl	-36(%rbp), %edx
	movq	head(%rip), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	display
	movq	$2, -24(%rbp)
	jmp	.L53
.L46:
	movl	-32(%rbp), %eax
	cmpl	$3, %eax
	je	.L54
	cmpl	$3, %eax
	jg	.L55
	cmpl	$1, %eax
	je	.L56
	cmpl	$2, %eax
	je	.L57
	jmp	.L55
.L54:
	movq	$8, -24(%rbp)
	jmp	.L58
.L57:
	movq	$18, -24(%rbp)
	jmp	.L58
.L56:
	movq	$15, -24(%rbp)
	jmp	.L58
.L55:
	movq	$1, -24(%rbp)
	nop
.L58:
	jmp	.L53
.L45:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	movq	head(%rip), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	input_node
	movq	$2, -24(%rbp)
	jmp	.L53
.L47:
	movl	$1, -28(%rbp)
	movl	$8, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, head(%rip)
	movq	$10, -24(%rbp)
	jmp	.L53
.L49:
	movq	head(%rip), %rax
	movq	%rax, %rdi
	call	insert_at_beginning
	movq	$2, -24(%rbp)
	jmp	.L53
.L51:
	movq	$2, -24(%rbp)
	jmp	.L53
.L44:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -24(%rbp)
	jmp	.L53
.L43:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L63
	jmp	.L64
.L48:
	movl	-28(%rbp), %eax
	testl	%eax, %eax
	je	.L60
	movq	$16, -24(%rbp)
	jmp	.L53
.L60:
	movq	$17, -24(%rbp)
	jmp	.L53
.L52:
	movq	$12, -24(%rbp)
	jmp	.L53
.L50:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$10, -24(%rbp)
	jmp	.L53
.L65:
	nop
.L53:
	jmp	.L62
.L64:
	call	__stack_chk_fail@PLT
.L63:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.section	.rodata
.LC7:
	.string	"%d ->"
.LC8:
	.string	"enter data you want to input"
.LC9:
	.string	"memory not allocated"
	.text
	.globl	insert_at_beginning
	.type	insert_at_beginning, @function
insert_at_beginning:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$4, -16(%rbp)
.L85:
	cmpq	$13, -16(%rbp)
	ja	.L86
	movq	-16(%rbp), %rax
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
	.long	.L86-.L69
	.long	.L78-.L69
	.long	.L77-.L69
	.long	.L76-.L69
	.long	.L75-.L69
	.long	.L74-.L69
	.long	.L73-.L69
	.long	.L72-.L69
	.long	.L86-.L69
	.long	.L71-.L69
	.long	.L70-.L69
	.long	.L86-.L69
	.long	.L86-.L69
	.long	.L87-.L69
	.text
.L75:
	movq	$7, -16(%rbp)
	jmp	.L79
.L78:
	movq	temp(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	temp(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, temp(%rip)
	movq	$10, -16(%rbp)
	jmp	.L79
.L76:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -16(%rbp)
	jmp	.L79
.L71:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L79
.L73:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L79
.L74:
	cmpq	$0, -24(%rbp)
	jne	.L81
	movq	$6, -16(%rbp)
	jmp	.L79
.L81:
	movq	$9, -16(%rbp)
	jmp	.L79
.L70:
	movq	temp(%rip), %rax
	testq	%rax, %rax
	je	.L83
	movq	$1, -16(%rbp)
	jmp	.L79
.L83:
	movq	$3, -16(%rbp)
	jmp	.L79
.L72:
	movl	$8, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L79
.L77:
	movq	-40(%rbp), %rax
	movq	%rax, temp(%rip)
	movq	$10, -16(%rbp)
	jmp	.L79
.L86:
	nop
.L79:
	jmp	.L85
.L87:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	insert_at_beginning, .-insert_at_beginning
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
