	.file	"tanishqraj360_DSA_4_flatten.c"
	.text
	.globl	ptr1
	.bss
	.align 8
	.type	ptr1, @object
	.size	ptr1, 8
ptr1:
	.zero	8
	.globl	_TIG_IZ_EMh8_envp
	.align 8
	.type	_TIG_IZ_EMh8_envp, @object
	.size	_TIG_IZ_EMh8_envp, 8
_TIG_IZ_EMh8_envp:
	.zero	8
	.globl	count
	.align 4
	.type	count, @object
	.size	count, 4
count:
	.zero	4
	.globl	_TIG_IZ_EMh8_argc
	.align 4
	.type	_TIG_IZ_EMh8_argc, @object
	.size	_TIG_IZ_EMh8_argc, 4
_TIG_IZ_EMh8_argc:
	.zero	4
	.globl	ptr
	.align 8
	.type	ptr, @object
	.size	ptr, 8
ptr:
	.zero	8
	.globl	last
	.align 8
	.type	last, @object
	.size	last, 8
last:
	.zero	8
	.globl	temp
	.align 8
	.type	temp, @object
	.size	temp, 8
temp:
	.zero	8
	.globl	_TIG_IZ_EMh8_argv
	.align 8
	.type	_TIG_IZ_EMh8_argv, @object
	.size	_TIG_IZ_EMh8_argv, 8
_TIG_IZ_EMh8_argv:
	.zero	8
	.globl	first
	.align 8
	.type	first, @object
	.size	first, 8
first:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"\nKey value does not exist in list"
.LC1:
	.string	"Data to be deleted: %d"
.LC2:
	.string	"Enter key value: "
.LC3:
	.string	"%d"
.LC4:
	.string	"\nList is empty"
	.text
	.globl	delete_at_key
	.type	delete_at_key, @function
delete_at_key:
.LFB3:
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
	movq	$6, -16(%rbp)
.L40:
	cmpq	$28, -16(%rbp)
	ja	.L43
	movq	-16(%rbp), %rax
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
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L43-.L4
	.long	.L43-.L4
	.long	.L43-.L4
	.long	.L43-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L43-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L43-.L4
	.long	.L43-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L44-.L4
	.long	.L5-.L4
	.long	.L43-.L4
	.long	.L43-.L4
	.long	.L3-.L4
	.text
.L12:
	movq	ptr1(%rip), %rax
	testq	%rax, %rax
	jne	.L24
	movq	$10, -16(%rbp)
	jmp	.L26
.L24:
	movq	$15, -16(%rbp)
	jmp	.L26
.L5:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -16(%rbp)
	jmp	.L26
.L16:
	movq	ptr(%rip), %rax
	movl	16(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ptr1(%rip), %rax
	movq	%rax, last(%rip)
	movq	last(%rip), %rax
	movq	$0, (%rax)
	movq	ptr(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -16(%rbp)
	jmp	.L26
.L15:
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L27
	movq	$14, -16(%rbp)
	jmp	.L26
.L27:
	movq	$23, -16(%rbp)
	jmp	.L26
.L22:
	movq	first(%rip), %rax
	movq	%rax, ptr(%rip)
	movq	$0, ptr1(%rip)
	movq	$28, -16(%rbp)
	jmp	.L26
.L7:
	movq	ptr(%rip), %rax
	movl	16(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	movq	ptr1(%rip), %rdx
	movq	%rdx, 8(%rax)
	movq	ptr(%rip), %rdx
	movq	ptr1(%rip), %rax
	movq	(%rdx), %rdx
	movq	%rdx, (%rax)
	movq	ptr(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -16(%rbp)
	jmp	.L26
.L14:
	movl	$0, -20(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$19, -16(%rbp)
	jmp	.L26
.L9:
	movq	ptr(%rip), %rax
	movl	16(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, first(%rip)
	movq	ptr(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -16(%rbp)
	jmp	.L26
.L17:
	movq	first(%rip), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L30
	movq	$21, -16(%rbp)
	jmp	.L26
.L30:
	movq	$18, -16(%rbp)
	jmp	.L26
.L19:
	movq	ptr(%rip), %rax
	movl	16(%rax), %edx
	movl	-24(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L32
	movq	$20, -16(%rbp)
	jmp	.L26
.L32:
	movq	$22, -16(%rbp)
	jmp	.L26
.L11:
	movq	ptr(%rip), %rax
	testq	%rax, %rax
	je	.L34
	movq	$9, -16(%rbp)
	jmp	.L26
.L34:
	movq	$0, -16(%rbp)
	jmp	.L26
.L13:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -16(%rbp)
	jmp	.L26
.L21:
	movq	$1, -16(%rbp)
	jmp	.L26
.L8:
	movq	ptr(%rip), %rax
	movq	%rax, ptr1(%rip)
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	movq	%rax, ptr(%rip)
	movq	$19, -16(%rbp)
	jmp	.L26
.L3:
	movq	ptr(%rip), %rax
	testq	%rax, %rax
	jne	.L36
	movq	$17, -16(%rbp)
	jmp	.L26
.L36:
	movq	$16, -16(%rbp)
	jmp	.L26
.L18:
	movq	ptr(%rip), %rax
	movl	16(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	movq	%rax, first(%rip)
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	movq	$0, 8(%rax)
	movq	ptr(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -16(%rbp)
	jmp	.L26
.L23:
	cmpl	$1, -20(%rbp)
	jne	.L38
	movq	$11, -16(%rbp)
	jmp	.L26
.L38:
	movq	$25, -16(%rbp)
	jmp	.L26
.L20:
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$24, -16(%rbp)
	jmp	.L26
.L10:
	movl	$1, -20(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L26
.L43:
	nop
.L26:
	jmp	.L40
.L44:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L42
	call	__stack_chk_fail@PLT
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	delete_at_key, .-delete_at_key
	.section	.rodata
.LC5:
	.string	"\n Enter no of nodes: "
.LC6:
	.string	"\n Enter choice : "
	.align 8
.LC7:
	.string	"-----------------MENU--------------------"
.LC8:
	.string	"\n 1 - Create DLL with N nodes"
.LC9:
	.string	"\n 2 - Display DLL"
.LC10:
	.string	"\n 3 - Delete at Key value"
.LC11:
	.string	"\n 4 - Insert at Key value"
.LC12:
	.string	"\n 5 - Exit"
	.align 8
.LC13:
	.string	"------------------------------------------"
.LC14:
	.string	"wrong choice"
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
	movl	$0, count(%rip)
	nop
.L46:
	movq	$0, ptr1(%rip)
	nop
.L47:
	movq	$0, ptr(%rip)
	nop
.L48:
	movq	$0, last(%rip)
	nop
.L49:
	movq	$0, temp(%rip)
	nop
.L50:
	movq	$0, first(%rip)
	nop
.L51:
	movq	$0, _TIG_IZ_EMh8_envp(%rip)
	nop
.L52:
	movq	$0, _TIG_IZ_EMh8_argv(%rip)
	nop
.L53:
	movl	$0, _TIG_IZ_EMh8_argc(%rip)
	nop
	nop
.L54:
.L55:
#APP
# 204 "tanishqraj360_DSA_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EMh8--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_EMh8_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_EMh8_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_EMh8_envp(%rip)
	nop
	movq	$10, -16(%rbp)
.L81:
	cmpq	$20, -16(%rbp)
	ja	.L83
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L58(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L58(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L58:
	.long	.L83-.L58
	.long	.L69-.L58
	.long	.L68-.L58
	.long	.L83-.L58
	.long	.L83-.L58
	.long	.L67-.L58
	.long	.L66-.L58
	.long	.L83-.L58
	.long	.L65-.L58
	.long	.L64-.L58
	.long	.L63-.L58
	.long	.L83-.L58
	.long	.L83-.L58
	.long	.L62-.L58
	.long	.L83-.L58
	.long	.L61-.L58
	.long	.L83-.L58
	.long	.L60-.L58
	.long	.L59-.L58
	.long	.L83-.L58
	.long	.L57-.L58
	.text
.L59:
	call	display
	movq	$17, -16(%rbp)
	jmp	.L70
.L61:
	call	insert_at_first
	addl	$1, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L70
.L65:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L70
.L69:
	movl	-28(%rbp), %eax
	cmpl	$5, %eax
	ja	.L71
	movl	%eax, %eax
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
	.long	.L71-.L73
	.long	.L77-.L73
	.long	.L76-.L73
	.long	.L75-.L73
	.long	.L74-.L73
	.long	.L72-.L73
	.text
.L72:
	movq	$13, -16(%rbp)
	jmp	.L78
.L74:
	movq	$6, -16(%rbp)
	jmp	.L78
.L75:
	movq	$2, -16(%rbp)
	jmp	.L78
.L76:
	movq	$18, -16(%rbp)
	jmp	.L78
.L77:
	movq	$8, -16(%rbp)
	jmp	.L78
.L71:
	movq	$20, -16(%rbp)
	nop
.L78:
	jmp	.L70
.L64:
	movl	-24(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L79
	movq	$15, -16(%rbp)
	jmp	.L70
.L79:
	movq	$17, -16(%rbp)
	jmp	.L70
.L62:
	movl	$0, %edi
	call	exit@PLT
.L60:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -16(%rbp)
	jmp	.L70
.L66:
	call	insert_at_key
	movq	$17, -16(%rbp)
	jmp	.L70
.L67:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
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
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L70
.L63:
	movq	$5, -16(%rbp)
	jmp	.L70
.L68:
	call	delete_at_key
	movq	$17, -16(%rbp)
	jmp	.L70
.L57:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L70
.L83:
	nop
.L70:
	jmp	.L81
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.globl	insert_at_key
	.type	insert_at_key, @function
insert_at_key:
.LFB5:
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
	movq	$10, -16(%rbp)
.L113:
	cmpq	$20, -16(%rbp)
	ja	.L116
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L87(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L87(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L87:
	.long	.L116-.L87
	.long	.L100-.L87
	.long	.L117-.L87
	.long	.L98-.L87
	.long	.L97-.L87
	.long	.L96-.L87
	.long	.L116-.L87
	.long	.L95-.L87
	.long	.L94-.L87
	.long	.L93-.L87
	.long	.L92-.L87
	.long	.L91-.L87
	.long	.L116-.L87
	.long	.L116-.L87
	.long	.L116-.L87
	.long	.L90-.L87
	.long	.L116-.L87
	.long	.L116-.L87
	.long	.L89-.L87
	.long	.L88-.L87
	.long	.L86-.L87
	.text
.L89:
	movq	ptr(%rip), %rax
	testq	%rax, %rax
	jne	.L101
	movq	$1, -16(%rbp)
	jmp	.L103
.L101:
	movq	$19, -16(%rbp)
	jmp	.L103
.L97:
	cmpl	$1, -20(%rbp)
	jne	.L104
	movq	$15, -16(%rbp)
	jmp	.L103
.L104:
	movq	$7, -16(%rbp)
	jmp	.L103
.L90:
	movq	ptr1(%rip), %rax
	testq	%rax, %rax
	jne	.L106
	movq	$9, -16(%rbp)
	jmp	.L103
.L106:
	movq	$5, -16(%rbp)
	jmp	.L103
.L94:
	movq	ptr(%rip), %rax
	testq	%rax, %rax
	je	.L108
	movq	$3, -16(%rbp)
	jmp	.L103
.L108:
	movq	$4, -16(%rbp)
	jmp	.L103
.L100:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L103
.L98:
	movq	ptr(%rip), %rax
	movl	16(%rax), %edx
	movl	-24(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L110
	movq	$11, -16(%rbp)
	jmp	.L103
.L110:
	movq	$20, -16(%rbp)
	jmp	.L103
.L91:
	movl	$1, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L103
.L93:
	call	create
	movq	temp(%rip), %rax
	movq	first(%rip), %rdx
	movq	%rdx, (%rax)
	movq	temp(%rip), %rax
	movq	$0, 8(%rax)
	movq	first(%rip), %rax
	movq	%rax, last(%rip)
	movq	temp(%rip), %rax
	movq	%rax, first(%rip)
	movq	$2, -16(%rbp)
	jmp	.L103
.L88:
	movl	$0, -20(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L103
.L96:
	call	create
	movq	temp(%rip), %rax
	movq	ptr(%rip), %rdx
	movq	%rdx, (%rax)
	movq	ptr(%rip), %rax
	movq	temp(%rip), %rdx
	movq	%rdx, 8(%rax)
	movq	ptr1(%rip), %rax
	movq	temp(%rip), %rdx
	movq	%rdx, (%rax)
	movq	temp(%rip), %rax
	movq	ptr1(%rip), %rdx
	movq	%rdx, 8(%rax)
	movq	$2, -16(%rbp)
	jmp	.L103
.L92:
	movq	first(%rip), %rax
	movq	%rax, ptr(%rip)
	movq	$18, -16(%rbp)
	jmp	.L103
.L95:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L103
.L86:
	movq	ptr(%rip), %rax
	movq	%rax, ptr1(%rip)
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	movq	%rax, ptr(%rip)
	movq	$8, -16(%rbp)
	jmp	.L103
.L116:
	nop
.L103:
	jmp	.L113
.L117:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L115
	call	__stack_chk_fail@PLT
.L115:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	insert_at_key, .-insert_at_key
	.section	.rodata
.LC15:
	.string	"Enter data for node: "
	.text
	.globl	create
	.type	create, @function
create:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -16(%rbp)
.L124:
	cmpq	$2, -16(%rbp)
	je	.L119
	cmpq	$2, -16(%rbp)
	ja	.L125
	cmpq	$0, -16(%rbp)
	je	.L126
	cmpq	$1, -16(%rbp)
	jne	.L125
	movq	$2, -16(%rbp)
	jmp	.L122
.L119:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, temp(%rip)
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	temp(%rip), %rax
	addq	$16, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	temp(%rip), %rax
	movq	$0, (%rax)
	movq	temp(%rip), %rax
	movq	$0, 8(%rax)
	movl	count(%rip), %eax
	addl	$1, %eax
	movl	%eax, count(%rip)
	movq	$0, -16(%rbp)
	jmp	.L122
.L125:
	nop
.L122:
	jmp	.L124
.L126:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	create, .-create
	.globl	insert_at_first
	.type	insert_at_first, @function
insert_at_first:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L137:
	cmpq	$5, -8(%rbp)
	je	.L128
	cmpq	$5, -8(%rbp)
	ja	.L138
	cmpq	$4, -8(%rbp)
	je	.L139
	cmpq	$4, -8(%rbp)
	ja	.L138
	cmpq	$1, -8(%rbp)
	je	.L131
	cmpq	$2, -8(%rbp)
	je	.L132
	jmp	.L138
.L131:
	movq	first(%rip), %rax
	testq	%rax, %rax
	jne	.L134
	movq	$5, -8(%rbp)
	jmp	.L136
.L134:
	movq	$2, -8(%rbp)
	jmp	.L136
.L128:
	call	create
	movq	temp(%rip), %rax
	movq	%rax, first(%rip)
	movq	first(%rip), %rax
	movq	%rax, last(%rip)
	movq	$4, -8(%rbp)
	jmp	.L136
.L132:
	call	create
	movq	temp(%rip), %rax
	movq	first(%rip), %rdx
	movq	%rdx, (%rax)
	movq	temp(%rip), %rax
	movq	$0, 8(%rax)
	movq	temp(%rip), %rax
	movq	%rax, first(%rip)
	movq	$4, -8(%rbp)
	jmp	.L136
.L138:
	nop
.L136:
	jmp	.L137
.L139:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	insert_at_first, .-insert_at_first
	.section	.rodata
.LC16:
	.string	"%d\t"
.LC17:
	.string	" No of nodes = %d"
.LC18:
	.string	"List is empty"
.LC19:
	.string	"\n Linked list elements: "
	.text
	.globl	display
	.type	display, @function
display:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$11, -8(%rbp)
.L158:
	cmpq	$11, -8(%rbp)
	ja	.L159
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L143(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L143(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L143:
	.long	.L151-.L143
	.long	.L159-.L143
	.long	.L150-.L143
	.long	.L149-.L143
	.long	.L148-.L143
	.long	.L147-.L143
	.long	.L159-.L143
	.long	.L160-.L143
	.long	.L145-.L143
	.long	.L159-.L143
	.long	.L160-.L143
	.long	.L142-.L143
	.text
.L148:
	movq	ptr(%rip), %rax
	testq	%rax, %rax
	jne	.L152
	movq	$0, -8(%rbp)
	jmp	.L154
.L152:
	movq	$2, -8(%rbp)
	jmp	.L154
.L145:
	movq	ptr(%rip), %rax
	movl	16(%rax), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	ptr(%rip), %rax
	movq	(%rax), %rax
	movq	%rax, ptr(%rip)
	movq	$3, -8(%rbp)
	jmp	.L154
.L149:
	movq	ptr(%rip), %rax
	testq	%rax, %rax
	je	.L155
	movq	$8, -8(%rbp)
	jmp	.L154
.L155:
	movq	$5, -8(%rbp)
	jmp	.L154
.L142:
	movq	first(%rip), %rax
	movq	%rax, ptr(%rip)
	movq	$4, -8(%rbp)
	jmp	.L154
.L147:
	movl	count(%rip), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L154
.L151:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L154
.L150:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L154
.L159:
	nop
.L154:
	jmp	.L158
.L160:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
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
