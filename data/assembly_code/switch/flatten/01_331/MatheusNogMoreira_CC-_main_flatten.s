	.file	"MatheusNogMoreira_CC-_main_flatten.c"
	.text
	.globl	_TIG_IZ_PHc1_argc
	.bss
	.align 4
	.type	_TIG_IZ_PHc1_argc, @object
	.size	_TIG_IZ_PHc1_argc, 4
_TIG_IZ_PHc1_argc:
	.zero	4
	.globl	max
	.align 4
	.type	max, @object
	.size	max, 4
max:
	.zero	4
	.globl	_TIG_IZ_PHc1_argv
	.align 8
	.type	_TIG_IZ_PHc1_argv, @object
	.size	_TIG_IZ_PHc1_argv, 8
_TIG_IZ_PHc1_argv:
	.zero	8
	.globl	array
	.align 8
	.type	array, @object
	.size	array, 8
array:
	.zero	8
	.globl	_TIG_IZ_PHc1_envp
	.align 8
	.type	_TIG_IZ_PHc1_envp, @object
	.size	_TIG_IZ_PHc1_envp, 8
_TIG_IZ_PHc1_envp:
	.zero	8
	.globl	size
	.align 4
	.type	size, @object
	.size	size, 4
size:
	.zero	4
	.text
	.globl	init_array
	.type	init_array, @function
init_array:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L10:
	cmpq	$6, -8(%rbp)
	je	.L2
	cmpq	$6, -8(%rbp)
	ja	.L12
	cmpq	$5, -8(%rbp)
	je	.L4
	cmpq	$5, -8(%rbp)
	ja	.L12
	cmpq	$1, -8(%rbp)
	je	.L5
	cmpq	$4, -8(%rbp)
	jne	.L12
	jmp	.L11
.L5:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L7
.L2:
	movl	max(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L8
	movq	$5, -8(%rbp)
	jmp	.L7
.L8:
	movq	$4, -8(%rbp)
	jmp	.L7
.L4:
	movq	array(%rip), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	array(%rip), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movq	$0, 8(%rax)
	addl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L7
.L12:
	nop
.L7:
	jmp	.L10
.L11:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	init_array, .-init_array
	.globl	hashcode
	.type	hashcode, @function
hashcode:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L16:
	cmpq	$0, -8(%rbp)
	jne	.L19
	movl	max(%rip), %ecx
	movl	-20(%rbp), %eax
	cltd
	idivl	%ecx
	movl	%edx, %eax
	jmp	.L18
.L19:
	nop
	jmp	.L16
.L18:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	hashcode, .-hashcode
	.section	.rodata
	.align 8
.LC0:
	.string	"Implementation of Hash Table in C with Linear Probing \n"
	.align 8
.LC1:
	.string	"MENU-: \n1.Inserting item in the Hashtable\n2.Removing item from the Hashtable\n3.Check the size of Hashtable\n4.Display Hashtable\n\n Please enter your choice-:"
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"Deleting in Hashtable \n Enter the key to delete-:"
.LC4:
	.string	"Size of Hashtable is-:%d\n"
	.align 8
.LC5:
	.string	"\n Do you want to continue-:(press 1 for yes)\t"
.LC6:
	.string	"Wrong Input"
	.align 8
.LC7:
	.string	"Inserting element in Hashtable"
.LC8:
	.string	"Enter key and value-:\t"
.LC9:
	.string	"%d %d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
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
	movl	$10, max(%rip)
	nop
.L21:
	movl	$0, size(%rip)
	nop
.L22:
	movq	$0, array(%rip)
	nop
.L23:
	movq	$0, _TIG_IZ_PHc1_envp(%rip)
	nop
.L24:
	movq	$0, _TIG_IZ_PHc1_argv(%rip)
	nop
.L25:
	movl	$0, _TIG_IZ_PHc1_argc(%rip)
	nop
	nop
.L26:
.L27:
#APP
# 129 "MatheusNogMoreira_CC-_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-PHc1--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_PHc1_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_PHc1_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_PHc1_envp(%rip)
	nop
	movq	$14, -24(%rbp)
.L52:
	cmpq	$22, -24(%rbp)
	ja	.L55
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L30(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L30(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L30:
	.long	.L55-.L30
	.long	.L55-.L30
	.long	.L41-.L30
	.long	.L55-.L30
	.long	.L40-.L30
	.long	.L55-.L30
	.long	.L39-.L30
	.long	.L38-.L30
	.long	.L37-.L30
	.long	.L36-.L30
	.long	.L55-.L30
	.long	.L35-.L30
	.long	.L55-.L30
	.long	.L55-.L30
	.long	.L34-.L30
	.long	.L33-.L30
	.long	.L55-.L30
	.long	.L32-.L30
	.long	.L55-.L30
	.long	.L55-.L30
	.long	.L55-.L30
	.long	.L31-.L30
	.long	.L29-.L30
	.text
.L40:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$22, -24(%rbp)
	jmp	.L42
.L34:
	movq	$21, -24(%rbp)
	jmp	.L42
.L33:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-40(%rbp), %eax
	movl	%eax, %edi
	call	remove_element
	movq	$9, -24(%rbp)
	jmp	.L42
.L37:
	call	size_of_hashtable
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -24(%rbp)
	jmp	.L42
.L31:
	movl	max(%rip), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, array(%rip)
	call	init_array
	movq	$4, -24(%rbp)
	jmp	.L42
.L35:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L53
	jmp	.L54
.L36:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -24(%rbp)
	jmp	.L42
.L32:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -24(%rbp)
	jmp	.L42
.L39:
	movl	-32(%rbp), %eax
	cmpl	$1, %eax
	jne	.L44
	movq	$4, -24(%rbp)
	jmp	.L42
.L44:
	movq	$11, -24(%rbp)
	jmp	.L42
.L29:
	movl	-44(%rbp), %eax
	cmpl	$4, %eax
	je	.L46
	cmpl	$4, %eax
	jg	.L47
	cmpl	$3, %eax
	je	.L48
	cmpl	$3, %eax
	jg	.L47
	cmpl	$1, %eax
	je	.L49
	cmpl	$2, %eax
	je	.L50
	jmp	.L47
.L46:
	movq	$7, -24(%rbp)
	jmp	.L51
.L48:
	movq	$8, -24(%rbp)
	jmp	.L51
.L50:
	movq	$15, -24(%rbp)
	jmp	.L51
.L49:
	movq	$2, -24(%rbp)
	jmp	.L51
.L47:
	movq	$17, -24(%rbp)
	nop
.L51:
	jmp	.L42
.L38:
	call	display
	movq	$9, -24(%rbp)
	jmp	.L42
.L41:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	movl	-40(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	insert
	movq	$9, -24(%rbp)
	jmp	.L42
.L55:
	nop
.L42:
	jmp	.L52
.L54:
	call	__stack_chk_fail@PLT
.L53:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"\n This key does not exist "
.LC11:
	.string	"\n Key (%d) has been removed \n"
	.text
	.globl	remove_element
	.type	remove_element, @function
remove_element:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$9, -8(%rbp)
.L80:
	cmpq	$13, -8(%rbp)
	ja	.L81
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L59(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L59(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L59:
	.long	.L81-.L59
	.long	.L69-.L59
	.long	.L68-.L59
	.long	.L67-.L59
	.long	.L82-.L59
	.long	.L65-.L59
	.long	.L81-.L59
	.long	.L82-.L59
	.long	.L63-.L59
	.long	.L62-.L59
	.long	.L61-.L59
	.long	.L81-.L59
	.long	.L60-.L59
	.long	.L58-.L59
	.text
.L60:
	movl	-36(%rbp), %eax
	movl	%eax, %edi
	call	hashcode
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L71
.L63:
	movq	array(%rip), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jne	.L72
	movq	$5, -8(%rbp)
	jmp	.L71
.L72:
	movq	$1, -8(%rbp)
	jmp	.L71
.L69:
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	max(%rip), %ecx
	cltd
	idivl	%ecx
	movl	%edx, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L71
.L67:
	movq	array(%rip), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L74
	movq	$10, -8(%rbp)
	jmp	.L71
.L74:
	movq	$13, -8(%rbp)
	jmp	.L71
.L62:
	movq	$12, -8(%rbp)
	jmp	.L71
.L58:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L71
.L65:
	movq	array(%rip), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movl	$2, (%rax)
	movq	array(%rip), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movq	$0, 8(%rax)
	movl	size(%rip), %eax
	subl	$1, %eax
	movl	%eax, size(%rip)
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L71
.L61:
	movq	array(%rip), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$1, %eax
	jne	.L76
	movq	$8, -8(%rbp)
	jmp	.L71
.L76:
	movq	$1, -8(%rbp)
	jmp	.L71
.L68:
	movl	-16(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jne	.L78
	movq	$13, -8(%rbp)
	jmp	.L71
.L78:
	movq	$3, -8(%rbp)
	jmp	.L71
.L81:
	nop
.L71:
	jmp	.L80
.L82:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	remove_element, .-remove_element
	.section	.rodata
	.align 8
.LC12:
	.string	"\n Hash table is full, cannot insert any more item "
	.align 8
.LC13:
	.string	"\n Key (%d) has been inserted \n"
	.align 8
.LC14:
	.string	"\n Key already exists, hence updating its value "
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movl	%esi, -56(%rbp)
	movq	$4, -16(%rbp)
.L106:
	cmpq	$13, -16(%rbp)
	ja	.L107
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L86(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L86(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L86:
	.long	.L97-.L86
	.long	.L107-.L86
	.long	.L96-.L86
	.long	.L95-.L86
	.long	.L94-.L86
	.long	.L108-.L86
	.long	.L92-.L86
	.long	.L91-.L86
	.long	.L90-.L86
	.long	.L108-.L86
	.long	.L107-.L86
	.long	.L88-.L86
	.long	.L87-.L86
	.long	.L108-.L86
	.text
.L94:
	movq	$8, -16(%rbp)
	jmp	.L98
.L87:
	movl	-32(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jne	.L99
	movq	$11, -16(%rbp)
	jmp	.L98
.L99:
	movq	$6, -16(%rbp)
	jmp	.L98
.L90:
	movl	-52(%rbp), %eax
	movl	%eax, %edi
	call	hashcode
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -32(%rbp)
	movl	$8, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-52(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movl	-56(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	$6, -16(%rbp)
	jmp	.L98
.L95:
	movq	array(%rip), %rdx
	movl	-32(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, -52(%rbp)
	jne	.L101
	movq	$2, -16(%rbp)
	jmp	.L98
.L101:
	movq	$7, -16(%rbp)
	jmp	.L98
.L88:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -16(%rbp)
	jmp	.L98
.L92:
	movq	array(%rip), %rdx
	movl	-32(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$1, %eax
	jne	.L104
	movq	$3, -16(%rbp)
	jmp	.L98
.L104:
	movq	$0, -16(%rbp)
	jmp	.L98
.L97:
	movq	array(%rip), %rdx
	movl	-32(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movl	$1, (%rax)
	movq	array(%rip), %rdx
	movl	-32(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, 8(%rdx)
	movl	size(%rip), %eax
	addl	$1, %eax
	movl	%eax, size(%rip)
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L98
.L91:
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	max(%rip), %ecx
	cltd
	idivl	%ecx
	movl	%edx, -32(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L98
.L96:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	array(%rip), %rdx
	movl	-32(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movq	8(%rax), %rax
	movl	-56(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	$9, -16(%rbp)
	jmp	.L98
.L107:
	nop
.L98:
	jmp	.L106
.L108:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	insert, .-insert
	.globl	size_of_hashtable
	.type	size_of_hashtable, @function
size_of_hashtable:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L112:
	cmpq	$0, -8(%rbp)
	jne	.L115
	movl	size(%rip), %eax
	jmp	.L114
.L115:
	nop
	jmp	.L112
.L114:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	size_of_hashtable, .-size_of_hashtable
	.section	.rodata
.LC15:
	.string	"\n Array[%d] has no elements \n"
	.align 8
.LC16:
	.string	"\n Array[%d] has elements -: \n  %d (key) and %d(value) "
	.text
	.globl	display
	.type	display, @function
display:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$4, -8(%rbp)
.L133:
	cmpq	$9, -8(%rbp)
	ja	.L134
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L119(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L119(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L119:
	.long	.L126-.L119
	.long	.L125-.L119
	.long	.L135-.L119
	.long	.L134-.L119
	.long	.L123-.L119
	.long	.L122-.L119
	.long	.L121-.L119
	.long	.L120-.L119
	.long	.L134-.L119
	.long	.L118-.L119
	.text
.L123:
	movl	$0, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L127
.L125:
	cmpq	$0, -16(%rbp)
	jne	.L128
	movq	$6, -8(%rbp)
	jmp	.L127
.L128:
	movq	$0, -8(%rbp)
	jmp	.L127
.L118:
	movq	array(%rip), %rdx
	movl	-20(%rbp), %eax
	cltq
	salq	$4, %rax
	addq	%rdx, %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L127
.L121:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L127
.L122:
	addl	$1, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L127
.L126:
	movq	-16(%rbp), %rax
	movl	4(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	(%rax), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L127
.L120:
	movl	max(%rip), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L130
	movq	$9, -8(%rbp)
	jmp	.L127
.L130:
	movq	$2, -8(%rbp)
	jmp	.L127
.L134:
	nop
.L127:
	jmp	.L133
.L135:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	display, .-display
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
