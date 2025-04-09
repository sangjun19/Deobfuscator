	.file	"SyntaxNova_OSL_1_flatten.c"
	.text
	.globl	_TIG_IZ_wRiz_argv
	.bss
	.align 8
	.type	_TIG_IZ_wRiz_argv, @object
	.size	_TIG_IZ_wRiz_argv, 8
_TIG_IZ_wRiz_argv:
	.zero	8
	.globl	_TIG_IZ_wRiz_envp
	.align 8
	.type	_TIG_IZ_wRiz_envp, @object
	.size	_TIG_IZ_wRiz_envp, 8
_TIG_IZ_wRiz_envp:
	.zero	8
	.globl	_TIG_IZ_wRiz_argc
	.align 4
	.type	_TIG_IZ_wRiz_argc, @object
	.size	_TIG_IZ_wRiz_argc, 4
_TIG_IZ_wRiz_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Name: %s, Phone: %s, Email: %s\n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB2:
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
.L11:
	cmpq	$6, -8(%rbp)
	je	.L2
	cmpq	$6, -8(%rbp)
	ja	.L12
	cmpq	$4, -8(%rbp)
	je	.L4
	cmpq	$4, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L13
	cmpq	$3, -8(%rbp)
	je	.L6
	jmp	.L12
.L4:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L7
.L6:
	movl	-12(%rbp), %eax
	cltq
	imulq	$115, %rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	leaq	65(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	imulq	$115, %rax, %rcx
	movq	-24(%rbp), %rax
	addq	%rcx, %rax
	addq	$50, %rax
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rcx
	imulq	$115, %rcx, %rsi
	movq	-24(%rbp), %rcx
	addq	%rsi, %rcx
	movq	%rcx, %rsi
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L7
.L2:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L8
	movq	$3, -8(%rbp)
	jmp	.L7
.L8:
	movq	$2, -8(%rbp)
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
.LFE2:
	.size	display, .-display
	.section	.rodata
.LC1:
	.string	"Enter Name to Delete: "
.LC2:
	.string	"\n"
.LC3:
	.string	"Enter Name to Modify: "
.LC4:
	.string	"Address Book Created."
.LC5:
	.string	"Enter Name: "
.LC6:
	.string	"Enter Phone: "
.LC7:
	.string	"Enter Email: "
	.align 8
.LC8:
	.string	"\n1. Create Address Book\n2. View Address Book\n3. Insert Record\n4. Delete Record\n5. Modify Record\n6. Exit"
.LC9:
	.string	"Enter choice: "
.LC10:
	.string	"%d"
.LC11:
	.string	"Invalid choice."
.LC12:
	.string	"Enter new Phone: "
.LC13:
	.string	"Enter new Email: "
.LC14:
	.string	"Record modified."
.LC15:
	.string	"Record deleted."
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
	pushq	%rbx
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$3512, %rsp
	.cfi_offset 3, -24
	movl	%edi, -11684(%rbp)
	movq	%rsi, -11696(%rbp)
	movq	%rdx, -11704(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_wRiz_envp(%rip)
	nop
.L15:
	movq	$0, _TIG_IZ_wRiz_argv(%rip)
	nop
.L16:
	movl	$0, _TIG_IZ_wRiz_argc(%rip)
	nop
	nop
.L17:
.L18:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wRiz--0
# 0 "" 2
#NO_APP
	movl	-11684(%rbp), %eax
	movl	%eax, _TIG_IZ_wRiz_argc(%rip)
	movq	-11696(%rbp), %rax
	movq	%rax, _TIG_IZ_wRiz_argv(%rip)
	movq	-11704(%rbp), %rax
	movq	%rax, _TIG_IZ_wRiz_envp(%rip)
	nop
	movq	$8, -11648(%rbp)
.L65:
	cmpq	$42, -11648(%rbp)
	ja	.L68
	movq	-11648(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L43-.L21
	.long	.L42-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L41-.L21
	.long	.L68-.L21
	.long	.L40-.L21
	.long	.L39-.L21
	.long	.L38-.L21
	.long	.L68-.L21
	.long	.L37-.L21
	.long	.L68-.L21
	.long	.L36-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L35-.L21
	.long	.L34-.L21
	.long	.L68-.L21
	.long	.L33-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L32-.L21
	.long	.L31-.L21
	.long	.L30-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L68-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L68-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L33:
	movl	-11668(%rbp), %eax
	movl	%eax, -11664(%rbp)
	movq	$23, -11648(%rbp)
	jmp	.L44
.L41:
	movl	-11668(%rbp), %eax
	cmpl	-11672(%rbp), %eax
	jge	.L45
	movq	$29, -11648(%rbp)
	jmp	.L44
.L45:
	movq	$27, -11648(%rbp)
	jmp	.L44
.L35:
	movl	-11672(%rbp), %edx
	leaq	-11584(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	display
	movq	$27, -11648(%rbp)
	jmp	.L44
.L36:
	cmpl	$0, -11652(%rbp)
	jne	.L47
	movq	$28, -11648(%rbp)
	jmp	.L44
.L47:
	movq	$10, -11648(%rbp)
	jmp	.L44
.L38:
	movl	$0, -11672(%rbp)
	movq	$27, -11648(%rbp)
	jmp	.L44
.L42:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-80(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-80(%rbp), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11592(%rbp)
	leaq	-80(%rbp), %rdx
	movq	-11592(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$0, -11668(%rbp)
	movq	$4, -11648(%rbp)
	jmp	.L44
.L30:
	movl	-11672(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -11664(%rbp)
	jge	.L49
	movq	$38, -11648(%rbp)
	jmp	.L44
.L49:
	movq	$41, -11648(%rbp)
	jmp	.L44
.L34:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-80(%rbp), %rax
	movl	$50, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-80(%rbp), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11624(%rbp)
	leaq	-80(%rbp), %rdx
	movq	-11624(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$0, -11656(%rbp)
	movq	$0, -11648(%rbp)
	jmp	.L44
.L32:
	movl	$0, -11672(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$27, -11648(%rbp)
	jmp	.L44
.L26:
	leaq	-11584(%rbp), %rdx
	movl	-11656(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	%rax, %rdx
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -11652(%rbp)
	movq	$12, -11648(%rbp)
	jmp	.L44
.L40:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	leaq	-11584(%rbp), %rcx
	movl	-11672(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$115, %rdx, %rdx
	addq	%rdx, %rcx
	movq	%rax, %rdx
	movl	$50, %esi
	movq	%rcx, %rdi
	call	fgets@PLT
	leaq	-11584(%rbp), %rdx
	movl	-11672(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	%rdx, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11616(%rbp)
	movl	-11672(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	leaq	-16(%rax), %rax
	leaq	(%rax,%rbp), %rdx
	movq	-11616(%rbp), %rax
	addq	%rdx, %rax
	subq	$11568, %rax
	movb	$0, (%rax)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	leaq	-11584(%rbp), %rcx
	movl	-11672(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$115, %rdx, %rdx
	addq	$48, %rdx
	addq	%rcx, %rdx
	leaq	2(%rdx), %rcx
	movq	%rax, %rdx
	movl	$15, %esi
	movq	%rcx, %rdi
	call	fgets@PLT
	leaq	-11584(%rbp), %rdx
	movl	-11672(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	$48, %rax
	addq	%rdx, %rax
	addq	$2, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11608(%rbp)
	movl	-11672(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	leaq	-16(%rax), %rax
	leaq	(%rax,%rbp), %rdx
	movq	-11608(%rbp), %rax
	addq	%rdx, %rax
	subq	$11518, %rax
	movb	$0, (%rax)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	leaq	-11584(%rbp), %rcx
	movl	-11672(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$115, %rdx, %rdx
	addq	$64, %rdx
	addq	%rcx, %rdx
	leaq	1(%rdx), %rcx
	movq	%rax, %rdx
	movl	$50, %esi
	movq	%rcx, %rdi
	call	fgets@PLT
	leaq	-11584(%rbp), %rdx
	movl	-11672(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	$64, %rax
	addq	%rdx, %rax
	addq	$1, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11600(%rbp)
	movl	-11672(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	leaq	-16(%rax), %rax
	leaq	(%rax,%rbp), %rdx
	movq	-11600(%rbp), %rax
	addq	%rdx, %rax
	subq	$11503, %rax
	movb	$0, (%rax)
	addl	$1, -11672(%rbp)
	movq	$27, -11648(%rbp)
	jmp	.L44
.L29:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-11676(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	getchar@PLT
	movq	$7, -11648(%rbp)
	jmp	.L44
.L24:
	movl	-11664(%rbp), %eax
	leal	1(%rax), %edx
	movl	-11664(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	leaq	-16(%rax), %rax
	addq	%rbp, %rax
	subq	$11568, %rax
	movslq	%edx, %rdx
	imulq	$115, %rdx, %rdx
	leaq	-16(%rdx), %rbx
	leaq	(%rbx,%rbp), %rdx
	subq	$11568, %rdx
	movq	(%rdx), %rcx
	movq	8(%rdx), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	16(%rdx), %rcx
	movq	24(%rdx), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	32(%rdx), %rcx
	movq	40(%rdx), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	48(%rdx), %rcx
	movq	56(%rdx), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	64(%rdx), %rcx
	movq	72(%rdx), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	80(%rdx), %rcx
	movq	88(%rdx), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	96(%rdx), %rcx
	movq	104(%rdx), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movzwl	112(%rdx), %ecx
	movw	%cx, 112(%rax)
	movzbl	114(%rdx), %edx
	movb	%dl, 114(%rax)
	addl	$1, -11664(%rbp)
	movq	$23, -11648(%rbp)
	jmp	.L44
.L31:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$27, -11648(%rbp)
	jmp	.L44
.L28:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	leaq	-11584(%rbp), %rcx
	movl	-11656(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$115, %rdx, %rdx
	addq	$48, %rdx
	addq	%rcx, %rdx
	leaq	2(%rdx), %rcx
	movq	%rax, %rdx
	movl	$15, %esi
	movq	%rcx, %rdi
	call	fgets@PLT
	leaq	-11584(%rbp), %rdx
	movl	-11656(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	$48, %rax
	addq	%rdx, %rax
	addq	$2, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11640(%rbp)
	movl	-11656(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	leaq	-16(%rax), %rax
	leaq	(%rax,%rbp), %rdx
	movq	-11640(%rbp), %rax
	addq	%rdx, %rax
	subq	$11518, %rax
	movb	$0, (%rax)
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rax
	leaq	-11584(%rbp), %rcx
	movl	-11656(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$115, %rdx, %rdx
	addq	$64, %rdx
	addq	%rcx, %rdx
	leaq	1(%rdx), %rcx
	movq	%rax, %rdx
	movl	$50, %esi
	movq	%rcx, %rdi
	call	fgets@PLT
	leaq	-11584(%rbp), %rdx
	movl	-11656(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	$64, %rax
	addq	%rdx, %rax
	addq	$1, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -11632(%rbp)
	movl	-11656(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	leaq	-16(%rax), %rax
	leaq	(%rax,%rbp), %rdx
	movq	-11632(%rbp), %rax
	addq	%rdx, %rax
	subq	$11503, %rax
	movb	$0, (%rax)
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$27, -11648(%rbp)
	jmp	.L44
.L25:
	addl	$1, -11668(%rbp)
	movq	$4, -11648(%rbp)
	jmp	.L44
.L22:
	subl	$1, -11672(%rbp)
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$27, -11648(%rbp)
	jmp	.L44
.L37:
	addl	$1, -11656(%rbp)
	movq	$0, -11648(%rbp)
	jmp	.L44
.L20:
	cmpl	$0, -11660(%rbp)
	jne	.L51
	movq	$18, -11648(%rbp)
	jmp	.L44
.L51:
	movq	$33, -11648(%rbp)
	jmp	.L44
.L43:
	movl	-11656(%rbp), %eax
	cmpl	-11672(%rbp), %eax
	jge	.L53
	movq	$32, -11648(%rbp)
	jmp	.L44
.L53:
	movq	$27, -11648(%rbp)
	jmp	.L44
.L23:
	movl	$0, %eax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L66
	jmp	.L67
.L39:
	movl	-11676(%rbp), %eax
	cmpl	$6, %eax
	ja	.L56
	movl	%eax, %eax
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
	.long	.L56-.L58
	.long	.L63-.L58
	.long	.L62-.L58
	.long	.L61-.L58
	.long	.L60-.L58
	.long	.L59-.L58
	.long	.L57-.L58
	.text
.L57:
	movq	$39, -11648(%rbp)
	jmp	.L64
.L59:
	movq	$16, -11648(%rbp)
	jmp	.L64
.L60:
	movq	$1, -11648(%rbp)
	jmp	.L64
.L61:
	movq	$6, -11648(%rbp)
	jmp	.L64
.L62:
	movq	$15, -11648(%rbp)
	jmp	.L64
.L63:
	movq	$21, -11648(%rbp)
	jmp	.L64
.L56:
	movq	$22, -11648(%rbp)
	nop
.L64:
	jmp	.L44
.L27:
	leaq	-11584(%rbp), %rdx
	movl	-11668(%rbp), %eax
	cltq
	imulq	$115, %rax, %rax
	addq	%rax, %rdx
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -11660(%rbp)
	movq	$42, -11648(%rbp)
	jmp	.L44
.L68:
	nop
.L44:
	jmp	.L65
.L67:
	call	__stack_chk_fail@PLT
.L66:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
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
