	.file	"ngohainnam_COS10007-Developing-Technical-Software_Week3_2c_flatten.c"
	.text
	.globl	_TIG_IZ_Dhpg_argv
	.bss
	.align 8
	.type	_TIG_IZ_Dhpg_argv, @object
	.size	_TIG_IZ_Dhpg_argv, 8
_TIG_IZ_Dhpg_argv:
	.zero	8
	.globl	_TIG_IZ_Dhpg_argc
	.align 4
	.type	_TIG_IZ_Dhpg_argc, @object
	.size	_TIG_IZ_Dhpg_argc, 4
_TIG_IZ_Dhpg_argc:
	.zero	4
	.globl	_TIG_IZ_Dhpg_envp
	.align 8
	.type	_TIG_IZ_Dhpg_envp, @object
	.size	_TIG_IZ_Dhpg_envp, 8
_TIG_IZ_Dhpg_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The list is \n"
.LC1:
	.string	"Student Name: %s \n"
.LC2:
	.string	"Course Name: %s \n"
.LC3:
	.string	"Student ID: %d \n"
.LC4:
	.string	"Course ID: %d \n"
	.text
	.globl	printList
	.type	printList, @function
printList:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
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
	cmpq	$2, -8(%rbp)
	je	.L6
	jmp	.L12
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L7
.L5:
	cmpq	$0, -24(%rbp)
	je	.L9
	movq	$2, -8(%rbp)
	jmp	.L7
.L9:
	movq	$3, -8(%rbp)
	jmp	.L7
.L6:
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	addq	$24, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movl	20(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movl	44(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L7
.L12:
	nop
.L7:
	jmp	.L11
.L13:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	printList, .-printList
	.section	.rodata
.LC5:
	.string	"No memory available."
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movl	%ecx, -76(%rbp)
	movl	%r8d, -80(%rbp)
	movq	$3, -16(%rbp)
.L39:
	cmpq	$15, -16(%rbp)
	ja	.L40
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L40-.L17
	.long	.L28-.L17
	.long	.L27-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L24-.L17
	.long	.L40-.L17
	.long	.L40-.L17
	.long	.L40-.L17
	.long	.L23-.L17
	.long	.L22-.L17
	.long	.L21-.L17
	.long	.L20-.L17
	.long	.L19-.L17
	.long	.L41-.L17
	.long	.L16-.L17
	.text
.L25:
	movq	-32(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 48(%rax)
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 48(%rax)
	movq	$14, -16(%rbp)
	jmp	.L29
.L16:
	movq	-24(%rbp), %rax
	movl	20(%rax), %eax
	cmpl	%eax, -76(%rbp)
	jle	.L31
	movq	$12, -16(%rbp)
	jmp	.L29
.L31:
	movq	$2, -16(%rbp)
	jmp	.L29
.L20:
	movq	-24(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L29
.L28:
	movl	$56, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L29
.L26:
	movq	$1, -16(%rbp)
	jmp	.L29
.L21:
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 48(%rax)
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$14, -16(%rbp)
	jmp	.L29
.L23:
	cmpq	$0, -40(%rbp)
	je	.L33
	movq	$10, -16(%rbp)
	jmp	.L29
.L33:
	movq	$13, -16(%rbp)
	jmp	.L29
.L19:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$14, -16(%rbp)
	jmp	.L29
.L24:
	cmpq	$0, -24(%rbp)
	je	.L35
	movq	$15, -16(%rbp)
	jmp	.L29
.L35:
	movq	$2, -16(%rbp)
	jmp	.L29
.L22:
	movq	-40(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-40(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-72(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcpy@PLT
	movq	-40(%rbp), %rax
	movl	-76(%rbp), %edx
	movl	%edx, 20(%rax)
	movq	-40(%rbp), %rax
	movl	-80(%rbp), %edx
	movl	%edx, 44(%rax)
	movq	-40(%rbp), %rax
	movq	$0, 48(%rax)
	movq	$0, -32(%rbp)
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L29
.L27:
	cmpq	$0, -32(%rbp)
	jne	.L37
	movq	$11, -16(%rbp)
	jmp	.L29
.L37:
	movq	$4, -16(%rbp)
	jmp	.L29
.L40:
	nop
.L29:
	jmp	.L39
.L41:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	insert, .-insert
	.section	.rodata
.LC6:
	.string	"w+"
.LC7:
	.string	"Output.txt"
	.text
	.globl	printfFile
	.type	printfFile, @function
printfFile:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$2, -16(%rbp)
.L54:
	cmpq	$7, -16(%rbp)
	ja	.L55
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
	.long	.L56-.L45
	.long	.L48-.L45
	.long	.L47-.L45
	.long	.L55-.L45
	.long	.L46-.L45
	.long	.L55-.L45
	.long	.L44-.L45
	.text
.L47:
	cmpq	$0, -40(%rbp)
	je	.L51
	movq	$7, -16(%rbp)
	jmp	.L53
.L51:
	movq	$1, -16(%rbp)
	jmp	.L53
.L46:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L53
.L44:
	movq	-40(%rbp), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	movl	20(%rax), %edx
	movq	-24(%rbp), %rax
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	movl	44(%rax), %edx
	movq	-24(%rbp), %rax
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L53
.L48:
	movq	$5, -16(%rbp)
	jmp	.L53
.L55:
	nop
.L53:
	jmp	.L54
.L56:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	printfFile, .-printfFile
	.section	.rodata
.LC8:
	.string	"r"
.LC9:
	.string	"student.txt"
.LC10:
	.string	"Error opening the file "
.LC11:
	.string	"%s"
.LC12:
	.string	"%d"
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
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Dhpg_envp(%rip)
	nop
.L58:
	movq	$0, _TIG_IZ_Dhpg_argv(%rip)
	nop
.L59:
	movl	$0, _TIG_IZ_Dhpg_argc(%rip)
	nop
	nop
.L60:
.L61:
#APP
# 128 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Dhpg--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_Dhpg_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_Dhpg_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_Dhpg_envp(%rip)
	nop
	movq	$1, -80(%rbp)
.L81:
	cmpq	$13, -80(%rbp)
	ja	.L84
	movq	-80(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L64(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L64(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L64:
	.long	.L74-.L64
	.long	.L73-.L64
	.long	.L72-.L64
	.long	.L71-.L64
	.long	.L70-.L64
	.long	.L84-.L64
	.long	.L69-.L64
	.long	.L68-.L64
	.long	.L84-.L64
	.long	.L67-.L64
	.long	.L84-.L64
	.long	.L66-.L64
	.long	.L65-.L64
	.long	.L63-.L64
	.text
.L70:
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -100(%rbp)
	movq	$9, -80(%rbp)
	jmp	.L75
.L65:
	movl	$0, %eax
	jmp	.L82
.L73:
	movq	$3, -80(%rbp)
	jmp	.L75
.L71:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	$6, -80(%rbp)
	jmp	.L75
.L66:
	movq	$0, -96(%rbp)
	movq	$4, -80(%rbp)
	jmp	.L75
.L67:
	cmpl	$0, -100(%rbp)
	je	.L77
	movq	$7, -80(%rbp)
	jmp	.L75
.L77:
	movq	$2, -80(%rbp)
	jmp	.L75
.L63:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -80(%rbp)
	jmp	.L75
.L69:
	cmpq	$0, -88(%rbp)
	jne	.L79
	movq	$13, -80(%rbp)
	jmp	.L75
.L79:
	movq	$11, -80(%rbp)
	jmp	.L75
.L74:
	movl	$-1, %eax
	jmp	.L82
.L68:
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	printList
	movq	$12, -80(%rbp)
	jmp	.L75
.L72:
	leaq	-64(%rbp), %rdx
	movq	-88(%rbp), %rax
	leaq	.LC11(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	leaq	-32(%rbp), %rdx
	movq	-88(%rbp), %rax
	leaq	.LC11(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	leaq	-108(%rbp), %rdx
	movq	-88(%rbp), %rax
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	leaq	-104(%rbp), %rdx
	movq	-88(%rbp), %rax
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	-104(%rbp), %edi
	movl	-108(%rbp), %ecx
	leaq	-32(%rbp), %rdx
	leaq	-64(%rbp), %rsi
	leaq	-96(%rbp), %rax
	movl	%edi, %r8d
	movq	%rax, %rdi
	call	insert
	movq	$4, -80(%rbp)
	jmp	.L75
.L84:
	nop
.L75:
	jmp	.L81
.L82:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L83
	call	__stack_chk_fail@PLT
.L83:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
