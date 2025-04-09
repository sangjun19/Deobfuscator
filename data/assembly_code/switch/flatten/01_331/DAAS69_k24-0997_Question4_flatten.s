	.file	"DAAS69_k24-0997_Question4_flatten.c"
	.text
	.globl	_TIG_IZ_oIgu_envp
	.bss
	.align 8
	.type	_TIG_IZ_oIgu_envp, @object
	.size	_TIG_IZ_oIgu_envp, 8
_TIG_IZ_oIgu_envp:
	.zero	8
	.globl	_TIG_IZ_oIgu_argc
	.align 4
	.type	_TIG_IZ_oIgu_argc, @object
	.size	_TIG_IZ_oIgu_argc, 4
_TIG_IZ_oIgu_argc:
	.zero	4
	.globl	_TIG_IZ_oIgu_argv
	.align 8
	.type	_TIG_IZ_oIgu_argv, @object
	.size	_TIG_IZ_oIgu_argv, 8
_TIG_IZ_oIgu_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"We're sorry but we dont have any book by that author"
.LC1:
	.string	"Book %d:\n"
.LC2:
	.string	"Title: %s\n"
.LC3:
	.string	"Author: %s\n"
.LC4:
	.string	"Publication Year: %d\n"
.LC5:
	.string	"Price: %d\n\n"
	.text
	.globl	list_book
	.type	list_book, @function
list_book:
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
	movq	%rsi, -48(%rbp)
	movq	$4, -8(%rbp)
.L22:
	cmpq	$11, -8(%rbp)
	ja	.L23
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
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L23-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L23-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L24-.L4
	.text
.L10:
	movq	$3, -8(%rbp)
	jmp	.L14
.L12:
	cmpl	$0, -12(%rbp)
	jne	.L15
	movq	$7, -8(%rbp)
	jmp	.L14
.L15:
	movq	$10, -8(%rbp)
	jmp	.L14
.L11:
	movl	$0, -20(%rbp)
	movl	$0, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L14
.L6:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	leaq	9(%rax), %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L14
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L14
.L9:
	cmpl	$0, -20(%rbp)
	jne	.L18
	movq	$6, -8(%rbp)
	jmp	.L14
.L18:
	movq	$11, -8(%rbp)
	jmp	.L14
.L5:
	addl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L14
.L13:
	cmpl	$2, -16(%rbp)
	jg	.L20
	movq	$9, -8(%rbp)
	jmp	.L14
.L20:
	movq	$5, -8(%rbp)
	jmp	.L14
.L7:
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	addq	$9, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	24(%rax), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L14
.L23:
	nop
.L14:
	jmp	.L22
.L24:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	list_book, .-list_book
	.section	.rodata
	.align 8
.LC6:
	.string	"We're sorry but we dont have any book by that title"
	.text
	.globl	search
	.type	search, @function
search:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$0, -8(%rbp)
.L46:
	cmpq	$11, -8(%rbp)
	ja	.L47
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L37-.L28
	.long	.L36-.L28
	.long	.L35-.L28
	.long	.L34-.L28
	.long	.L47-.L28
	.long	.L33-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L30-.L28
	.long	.L47-.L28
	.long	.L29-.L28
	.long	.L48-.L28
	.text
.L30:
	cmpl	$2, -16(%rbp)
	jg	.L38
	movq	$7, -8(%rbp)
	jmp	.L40
.L38:
	movq	$10, -8(%rbp)
	jmp	.L40
.L36:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L40
.L34:
	addl	$1, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L40
.L32:
	cmpl	$0, -12(%rbp)
	jne	.L42
	movq	$2, -8(%rbp)
	jmp	.L40
.L42:
	movq	$3, -8(%rbp)
	jmp	.L40
.L33:
	movl	$0, -20(%rbp)
	movl	$0, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L40
.L29:
	cmpl	$0, -20(%rbp)
	jne	.L44
	movq	$1, -8(%rbp)
	jmp	.L40
.L44:
	movq	$11, -8(%rbp)
	jmp	.L40
.L37:
	movq	$5, -8(%rbp)
	jmp	.L40
.L31:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L40
.L35:
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	addq	$9, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	24(%rax), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -20(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L40
.L47:
	nop
.L40:
	jmp	.L46
.L48:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	search, .-search
	.section	.rodata
	.align 8
.LC7:
	.string	"Please enter the title of the book: "
.LC8:
	.string	"%s"
	.align 8
.LC9:
	.string	"Please enter the author name: "
.LC10:
	.string	"Invalid choice"
	.align 8
.LC11:
	.string	"welcome to the library\nList all books'1'\nSearch books by author name'2'\nSearch book by title '3'"
.LC12:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_oIgu_envp(%rip)
	nop
.L50:
	movq	$0, _TIG_IZ_oIgu_argv(%rip)
	nop
.L51:
	movl	$0, _TIG_IZ_oIgu_argc(%rip)
	nop
	nop
.L52:
.L53:
#APP
# 87 "DAAS69_k24-0997_Question4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-oIgu--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_oIgu_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_oIgu_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_oIgu_envp(%rip)
	nop
	movq	$46, -120(%rbp)
.L101:
	cmpq	$54, -120(%rbp)
	ja	.L104
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L56(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L56(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L56:
	.long	.L81-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L80-.L56
	.long	.L79-.L56
	.long	.L78-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L77-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L76-.L56
	.long	.L104-.L56
	.long	.L75-.L56
	.long	.L74-.L56
	.long	.L73-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L72-.L56
	.long	.L71-.L56
	.long	.L70-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L69-.L56
	.long	.L68-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L67-.L56
	.long	.L104-.L56
	.long	.L66-.L56
	.long	.L104-.L56
	.long	.L65-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L104-.L56
	.long	.L64-.L56
	.long	.L104-.L56
	.long	.L63-.L56
	.long	.L62-.L56
	.long	.L104-.L56
	.long	.L61-.L56
	.long	.L60-.L56
	.long	.L59-.L56
	.long	.L58-.L56
	.long	.L57-.L56
	.long	.L55-.L56
	.text
.L74:
	cmpl	$8, -144(%rbp)
	jbe	.L82
	movq	$40, -120(%rbp)
	jmp	.L84
.L82:
	movq	$54, -120(%rbp)
	jmp	.L84
.L60:
	cmpl	$8, -128(%rbp)
	jbe	.L85
	movq	$29, -120(%rbp)
	jmp	.L84
.L85:
	movq	$11, -120(%rbp)
	jmp	.L84
.L61:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L102
	jmp	.L103
.L58:
	movl	-136(%rbp), %eax
	movb	$0, -84(%rbp,%rax)
	addl	$1, -136(%rbp)
	movq	$22, -120(%rbp)
	jmp	.L84
.L79:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	display
	movq	$49, -120(%rbp)
	jmp	.L84
.L68:
	movl	-148(%rbp), %eax
	cmpl	$3, %eax
	je	.L88
	cmpl	$3, %eax
	jg	.L89
	cmpl	$1, %eax
	je	.L90
	cmpl	$2, %eax
	je	.L91
	jmp	.L89
.L88:
	movq	$15, -120(%rbp)
	jmp	.L92
.L91:
	movq	$17, -120(%rbp)
	jmp	.L92
.L90:
	movq	$4, -120(%rbp)
	jmp	.L92
.L89:
	movq	$38, -120(%rbp)
	nop
.L92:
	jmp	.L84
.L76:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-28(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	search
	movq	$49, -120(%rbp)
	jmp	.L84
.L55:
	movl	-144(%rbp), %eax
	movb	$0, -112(%rbp,%rax)
	addl	$1, -144(%rbp)
	movq	$18, -120(%rbp)
	jmp	.L84
.L71:
	movl	-140(%rbp), %eax
	movb	$0, -103(%rbp,%rax)
	addl	$1, -140(%rbp)
	movq	$36, -120(%rbp)
	jmp	.L84
.L80:
	movb	$98, -112(%rbp)
	movb	$111, -111(%rbp)
	movb	$111, -110(%rbp)
	movb	$107, -109(%rbp)
	movb	$49, -108(%rbp)
	movb	$0, -107(%rbp)
	movl	$6, -144(%rbp)
	movq	$18, -120(%rbp)
	jmp	.L84
.L70:
	movb	$115, -75(%rbp)
	movb	$97, -74(%rbp)
	movb	$97, -73(%rbp)
	movb	$100, -72(%rbp)
	movb	$0, -71(%rbp)
	movl	$5, -132(%rbp)
	movq	$5, -120(%rbp)
	jmp	.L84
.L67:
	cmpl	$8, -140(%rbp)
	jbe	.L93
	movq	$47, -120(%rbp)
	jmp	.L84
.L93:
	movq	$23, -120(%rbp)
	jmp	.L84
.L77:
	movl	-128(%rbp), %eax
	movb	$0, -56(%rbp,%rax)
	addl	$1, -128(%rbp)
	movq	$50, -120(%rbp)
	jmp	.L84
.L59:
	movl	-132(%rbp), %eax
	movb	$0, -75(%rbp,%rax)
	addl	$1, -132(%rbp)
	movq	$5, -120(%rbp)
	jmp	.L84
.L73:
	cmpl	$8, -124(%rbp)
	jbe	.L95
	movq	$0, -120(%rbp)
	jmp	.L84
.L95:
	movq	$53, -120(%rbp)
	jmp	.L84
.L75:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-18(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-18(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	list_book
	movq	$49, -120(%rbp)
	jmp	.L84
.L65:
	movb	$97, -103(%rbp)
	movb	$104, -102(%rbp)
	movb	$109, -101(%rbp)
	movb	$101, -100(%rbp)
	movb	$100, -99(%rbp)
	movb	$0, -98(%rbp)
	movl	$6, -140(%rbp)
	movq	$36, -120(%rbp)
	jmp	.L84
.L66:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$49, -120(%rbp)
	jmp	.L84
.L72:
	cmpl	$8, -136(%rbp)
	jbe	.L97
	movq	$24, -120(%rbp)
	jmp	.L84
.L97:
	movq	$52, -120(%rbp)
	jmp	.L84
.L57:
	movl	-124(%rbp), %eax
	movb	$0, -47(%rbp,%rax)
	addl	$1, -124(%rbp)
	movq	$19, -120(%rbp)
	jmp	.L84
.L62:
	movl	$2019, -92(%rbp)
	movl	$2000, -88(%rbp)
	movb	$98, -84(%rbp)
	movb	$111, -83(%rbp)
	movb	$111, -82(%rbp)
	movb	$107, -81(%rbp)
	movb	$50, -80(%rbp)
	movb	$0, -79(%rbp)
	movl	$6, -136(%rbp)
	movq	$22, -120(%rbp)
	jmp	.L84
.L64:
	movl	$2021, -64(%rbp)
	movl	$2000, -60(%rbp)
	movb	$98, -56(%rbp)
	movb	$111, -55(%rbp)
	movb	$111, -54(%rbp)
	movb	$107, -53(%rbp)
	movb	$51, -52(%rbp)
	movb	$0, -51(%rbp)
	movl	$6, -128(%rbp)
	movq	$50, -120(%rbp)
	jmp	.L84
.L78:
	cmpl	$8, -132(%rbp)
	jbe	.L99
	movq	$44, -120(%rbp)
	jmp	.L84
.L99:
	movq	$51, -120(%rbp)
	jmp	.L84
.L81:
	movl	$2000, -36(%rbp)
	movl	$4000, -32(%rbp)
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-148(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$30, -120(%rbp)
	jmp	.L84
.L63:
	movq	$3, -120(%rbp)
	jmp	.L84
.L69:
	movb	$115, -47(%rbp)
	movb	$97, -46(%rbp)
	movb	$97, -45(%rbp)
	movb	$100, -44(%rbp)
	movb	$0, -43(%rbp)
	movl	$5, -124(%rbp)
	movq	$19, -120(%rbp)
	jmp	.L84
.L104:
	nop
.L84:
	jmp	.L101
.L103:
	call	__stack_chk_fail@PLT
.L102:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
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
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$5, -8(%rbp)
.L115:
	cmpq	$5, -8(%rbp)
	je	.L106
	cmpq	$5, -8(%rbp)
	ja	.L116
	cmpq	$3, -8(%rbp)
	je	.L108
	cmpq	$3, -8(%rbp)
	ja	.L116
	cmpq	$1, -8(%rbp)
	je	.L109
	cmpq	$2, -8(%rbp)
	je	.L117
	jmp	.L116
.L109:
	cmpl	$2, -12(%rbp)
	jg	.L111
	movq	$3, -8(%rbp)
	jmp	.L113
.L111:
	movq	$2, -8(%rbp)
	jmp	.L113
.L108:
	movl	-12(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	addq	$9, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$3, %rax
	subq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	24(%rax), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L113
.L106:
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L113
.L116:
	nop
.L113:
	jmp	.L115
.L117:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
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
