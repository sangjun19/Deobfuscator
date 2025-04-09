	.file	"vidhi448_Linux-Shell_Main_flatten.c"
	.text
	.local	history
	.comm	history,8000,32
	.globl	_TIG_IZ_xoHD_argv
	.bss
	.align 8
	.type	_TIG_IZ_xoHD_argv, @object
	.size	_TIG_IZ_xoHD_argv, 8
_TIG_IZ_xoHD_argv:
	.zero	8
	.globl	_TIG_IZ_xoHD_argc
	.align 4
	.type	_TIG_IZ_xoHD_argc, @object
	.size	_TIG_IZ_xoHD_argc, 4
_TIG_IZ_xoHD_argc:
	.zero	4
	.globl	_TIG_IZ_xoHD_envp
	.align 8
	.type	_TIG_IZ_xoHD_envp, @object
	.size	_TIG_IZ_xoHD_envp, 8
_TIG_IZ_xoHD_envp:
	.zero	8
	.local	history_count
	.comm	history_count,4,4
	.section	.rodata
.LC0:
	.string	"%s"
	.text
	.globl	printDir2
	.type	printDir2, @function
printDir2:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1056, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -1048(%rbp)
.L7:
	cmpq	$2, -1048(%rbp)
	je	.L2
	cmpq	$2, -1048(%rbp)
	ja	.L10
	cmpq	$0, -1048(%rbp)
	je	.L11
	cmpq	$1, -1048(%rbp)
	jne	.L10
	movq	$2, -1048(%rbp)
	jmp	.L5
.L2:
	leaq	-1040(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	getcwd@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -1048(%rbp)
	jmp	.L5
.L10:
	nop
.L5:
	jmp	.L7
.L11:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	printDir2, .-printDir2
	.section	.rodata
	.align 8
.LC1:
	.string	"\nexit: exit [n]\nExit the shell.\nExits the shell with a status of N.  If N is omitted, the exit status is that of the last command executed.\n"
	.text
	.globl	helpexit
	.type	helpexit, @function
helpexit:
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
.L17:
	cmpq	$0, -8(%rbp)
	je	.L18
	cmpq	$1, -8(%rbp)
	jne	.L19
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L15
.L19:
	nop
.L15:
	jmp	.L17
.L18:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	helpexit, .-helpexit
	.section	.rodata
.LC2:
	.string	"parent:error\n"
.LC3:
	.string	"date"
.LC4:
	.string	"Command not found"
.LC5:
	.string	"ls"
.LC6:
	.string	"cat"
.LC7:
	.string	"fork creation failed!!!\n"
.LC8:
	.string	"rm"
.LC9:
	.string	"mkdir"
	.text
	.globl	external_commands
	.type	external_commands, @function
external_commands:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$1104, %rsp
	movq	%rdi, -5192(%rbp)
	movq	%rsi, -5200(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$18, -5144(%rbp)
.L65:
	cmpq	$28, -5144(%rbp)
	ja	.L68
	movq	-5144(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L68-.L23
	.long	.L46-.L23
	.long	.L45-.L23
	.long	.L44-.L23
	.long	.L43-.L23
	.long	.L42-.L23
	.long	.L68-.L23
	.long	.L41-.L23
	.long	.L40-.L23
	.long	.L39-.L23
	.long	.L38-.L23
	.long	.L68-.L23
	.long	.L37-.L23
	.long	.L36-.L23
	.long	.L35-.L23
	.long	.L34-.L23
	.long	.L33-.L23
	.long	.L32-.L23
	.long	.L31-.L23
	.long	.L30-.L23
	.long	.L68-.L23
	.long	.L69-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L68-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L31:
	call	fork@PLT
	movl	%eax, -5172(%rbp)
	movq	$5, -5144(%rbp)
	jmp	.L47
.L25:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L43:
	movq	-5192(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -5160(%rbp)
	movq	$13, -5144(%rbp)
	jmp	.L47
.L35:
	movq	-5200(%rbp), %rdx
	leaq	-3088(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-3088(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	leaq	-3088(%rbp), %rax
	addq	%rdx, %rax
	movl	$1952539695, (%rax)
	movw	$101, 4(%rax)
	movq	-5192(%rbp), %rdx
	leaq	-3088(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L34:
	cmpl	$0, -5152(%rbp)
	jne	.L48
	movq	$3, -5144(%rbp)
	jmp	.L47
.L48:
	movq	$19, -5144(%rbp)
	jmp	.L47
.L37:
	movq	-5200(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	leaq	-1040(%rbp), %rax
	addq	%rdx, %rax
	movl	$1684761903, (%rax)
	movw	$29289, 4(%rax)
	movb	$0, 6(%rax)
	movq	-5192(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L40:
	cmpl	$0, -5172(%rbp)
	jne	.L50
	movq	$24, -5144(%rbp)
	jmp	.L47
.L50:
	movq	$22, -5144(%rbp)
	jmp	.L47
.L46:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L27:
	cmpl	$0, -5168(%rbp)
	jne	.L52
	movq	$12, -5144(%rbp)
	jmp	.L47
.L52:
	movq	$1, -5144(%rbp)
	jmp	.L47
.L44:
	movq	-5200(%rbp), %rdx
	leaq	-5136(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-5136(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	leaq	-5136(%rbp), %rax
	addq	%rdx, %rax
	movl	$7564335, (%rax)
	movq	-5192(%rbp), %rdx
	leaq	-5136(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L33:
	movq	-5200(%rbp), %rdx
	leaq	-4112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-4112(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	leaq	-4112(%rbp), %rax
	addq	%rdx, %rax
	movl	$1952539439, (%rax)
	movb	$0, 4(%rax)
	movq	-5192(%rbp), %rdx
	leaq	-4112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L26:
	movq	-5192(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -5152(%rbp)
	movq	$15, -5144(%rbp)
	jmp	.L47
.L39:
	cmpl	$0, -5164(%rbp)
	jne	.L55
	movq	$17, -5144(%rbp)
	jmp	.L47
.L55:
	movq	$2, -5144(%rbp)
	jmp	.L47
.L36:
	cmpl	$0, -5160(%rbp)
	jne	.L57
	movq	$14, -5144(%rbp)
	jmp	.L47
.L57:
	movq	$28, -5144(%rbp)
	jmp	.L47
.L30:
	movq	-5192(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -5156(%rbp)
	movq	$7, -5144(%rbp)
	jmp	.L47
.L32:
	movq	-5200(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-2064(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	leaq	-2064(%rbp), %rax
	addq	%rdx, %rax
	movl	$7172655, (%rax)
	movq	-5192(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L24:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$21, -5144(%rbp)
	jmp	.L47
.L28:
	leaq	-5176(%rbp), %rcx
	movl	-5172(%rbp), %eax
	movl	$0, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	waitpid@PLT
	movl	%eax, -5148(%rbp)
	movq	$10, -5144(%rbp)
	jmp	.L47
.L22:
	movq	-5192(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -5164(%rbp)
	movq	$9, -5144(%rbp)
	jmp	.L47
.L42:
	cmpl	$0, -5172(%rbp)
	jns	.L59
	movq	$27, -5144(%rbp)
	jmp	.L47
.L59:
	movq	$8, -5144(%rbp)
	jmp	.L47
.L38:
	cmpl	$-1, -5148(%rbp)
	jne	.L61
	movq	$25, -5144(%rbp)
	jmp	.L47
.L61:
	movq	$21, -5144(%rbp)
	jmp	.L47
.L41:
	cmpl	$0, -5156(%rbp)
	jne	.L63
	movq	$16, -5144(%rbp)
	jmp	.L47
.L63:
	movq	$4, -5144(%rbp)
	jmp	.L47
.L45:
	movq	-5192(%rbp), %rax
	movq	(%rax), %rax
	leaq	.LC9(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -5168(%rbp)
	movq	$23, -5144(%rbp)
	jmp	.L47
.L68:
	nop
.L47:
	jmp	.L65
.L69:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L67
	call	__stack_chk_fail@PLT
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	external_commands, .-external_commands
	.section	.rodata
.LC10:
	.string	"readline"
	.text
	.globl	read_line
	.type	read_line, @function
read_line:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -16(%rbp)
.L87:
	cmpq	$8, -16(%rbp)
	ja	.L90
	movq	-16(%rbp), %rax
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
	.long	.L80-.L73
	.long	.L79-.L73
	.long	.L78-.L73
	.long	.L77-.L73
	.long	.L76-.L73
	.long	.L75-.L73
	.long	.L90-.L73
	.long	.L74-.L73
	.long	.L72-.L73
	.text
.L76:
	movq	$5, -16(%rbp)
	jmp	.L81
.L72:
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L88
	jmp	.L89
.L79:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L77:
	cmpq	$-1, -24(%rbp)
	jne	.L83
	movq	$2, -16(%rbp)
	jmp	.L81
.L83:
	movq	$8, -16(%rbp)
	jmp	.L81
.L75:
	movq	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	movq	stdin(%rip), %rdx
	leaq	-32(%rbp), %rcx
	leaq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	getline@PLT
	movq	%rax, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L81
.L80:
	cmpl	$0, -44(%rbp)
	je	.L85
	movq	$7, -16(%rbp)
	jmp	.L81
.L85:
	movq	$1, -16(%rbp)
	jmp	.L81
.L74:
	movl	$0, %edi
	call	exit@PLT
.L78:
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -44(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L81
.L90:
	nop
.L81:
	jmp	.L87
.L89:
	call	__stack_chk_fail@PLT
.L88:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	read_line, .-read_line
	.section	.rodata
	.align 8
.LC11:
	.string	"NO such Option or file or directory"
.LC12:
	.string	"-n"
.LC13:
	.string	"-P"
.LC14:
	.string	"--help"
.LC15:
	.string	"%s "
.LC16:
	.string	"cd"
.LC17:
	.string	"echo"
.LC18:
	.string	"history"
.LC19:
	.string	"pwd"
.LC20:
	.string	"exit"
.LC21:
	.string	"-d"
.LC22:
	.string	"Option not found"
.LC23:
	.string	"%d %s"
.LC24:
	.string	"-L"
.LC25:
	.string	"Argument required"
.LC26:
	.string	"Offset entered is wrong"
.LC27:
	.string	"Option not present"
.LC28:
	.string	"-c"
.LC29:
	.string	"NO such file or directory"
.LC30:
	.string	"-E"
	.text
	.globl	ownCmdHandler
	.type	ownCmdHandler, @function
ownCmdHandler:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -136(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$61, -24(%rbp)
.L238:
	cmpq	$100, -24(%rbp)
	ja	.L240
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L94(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L94(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L94:
	.long	.L172-.L94
	.long	.L240-.L94
	.long	.L171-.L94
	.long	.L240-.L94
	.long	.L170-.L94
	.long	.L240-.L94
	.long	.L169-.L94
	.long	.L168-.L94
	.long	.L167-.L94
	.long	.L166-.L94
	.long	.L165-.L94
	.long	.L164-.L94
	.long	.L240-.L94
	.long	.L163-.L94
	.long	.L162-.L94
	.long	.L161-.L94
	.long	.L160-.L94
	.long	.L159-.L94
	.long	.L158-.L94
	.long	.L157-.L94
	.long	.L156-.L94
	.long	.L240-.L94
	.long	.L155-.L94
	.long	.L240-.L94
	.long	.L154-.L94
	.long	.L153-.L94
	.long	.L240-.L94
	.long	.L152-.L94
	.long	.L240-.L94
	.long	.L151-.L94
	.long	.L150-.L94
	.long	.L149-.L94
	.long	.L148-.L94
	.long	.L147-.L94
	.long	.L146-.L94
	.long	.L145-.L94
	.long	.L240-.L94
	.long	.L144-.L94
	.long	.L143-.L94
	.long	.L142-.L94
	.long	.L240-.L94
	.long	.L240-.L94
	.long	.L240-.L94
	.long	.L240-.L94
	.long	.L141-.L94
	.long	.L140-.L94
	.long	.L139-.L94
	.long	.L138-.L94
	.long	.L137-.L94
	.long	.L136-.L94
	.long	.L135-.L94
	.long	.L134-.L94
	.long	.L133-.L94
	.long	.L132-.L94
	.long	.L131-.L94
	.long	.L130-.L94
	.long	.L129-.L94
	.long	.L240-.L94
	.long	.L128-.L94
	.long	.L127-.L94
	.long	.L126-.L94
	.long	.L125-.L94
	.long	.L124-.L94
	.long	.L123-.L94
	.long	.L122-.L94
	.long	.L121-.L94
	.long	.L120-.L94
	.long	.L119-.L94
	.long	.L118-.L94
	.long	.L117-.L94
	.long	.L116-.L94
	.long	.L115-.L94
	.long	.L240-.L94
	.long	.L240-.L94
	.long	.L114-.L94
	.long	.L113-.L94
	.long	.L112-.L94
	.long	.L111-.L94
	.long	.L240-.L94
	.long	.L240-.L94
	.long	.L110-.L94
	.long	.L109-.L94
	.long	.L240-.L94
	.long	.L240-.L94
	.long	.L108-.L94
	.long	.L107-.L94
	.long	.L240-.L94
	.long	.L106-.L94
	.long	.L105-.L94
	.long	.L104-.L94
	.long	.L103-.L94
	.long	.L102-.L94
	.long	.L101-.L94
	.long	.L100-.L94
	.long	.L99-.L94
	.long	.L98-.L94
	.long	.L240-.L94
	.long	.L97-.L94
	.long	.L96-.L94
	.long	.L95-.L94
	.long	.L93-.L94
	.text
.L158:
	movl	$10, %edi
	call	putchar@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L135:
	call	helpexit
	movq	$17, -24(%rbp)
	jmp	.L173
.L110:
	call	printDir2
	movl	$10, %edi
	call	putchar@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L153:
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L136:
	movl	-76(%rbp), %eax
	subl	$1, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	history(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$95, -24(%rbp)
	jmp	.L173
.L133:
	call	helppwd
	movq	$38, -24(%rbp)
	jmp	.L173
.L170:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L150:
	movq	-136(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, -76(%rbp)
	movq	$62, -24(%rbp)
	jmp	.L173
.L124:
	movl	history_count(%rip), %eax
	cmpl	%eax, -76(%rbp)
	jl	.L174
	movq	$93, -24(%rbp)
	jmp	.L173
.L174:
	movq	$49, -24(%rbp)
	jmp	.L173
.L162:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC12(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -84(%rbp)
	movq	$97, -24(%rbp)
	jmp	.L173
.L161:
	movq	$13, -24(%rbp)
	jmp	.L173
.L104:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L176
	movq	$25, -24(%rbp)
	jmp	.L173
.L176:
	movq	$88, -24(%rbp)
	jmp	.L173
.L129:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L178
	movq	$80, -24(%rbp)
	jmp	.L173
.L178:
	movq	$48, -24(%rbp)
	jmp	.L173
.L149:
	cmpl	$0, -68(%rbp)
	jne	.L180
	movq	$34, -24(%rbp)
	jmp	.L173
.L180:
	movq	$55, -24(%rbp)
	jmp	.L173
.L117:
	movl	history_count(%rip), %eax
	subl	$1, %eax
	movl	%eax, history_count(%rip)
	movq	$38, -24(%rbp)
	jmp	.L173
.L167:
	cmpl	$0, -116(%rbp)
	jne	.L182
	movq	$75, -24(%rbp)
	jmp	.L173
.L182:
	movq	$54, -24(%rbp)
	jmp	.L173
.L140:
	movl	-76(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-76(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	history(%rip), %rax
	movq	(%rdx,%rax), %rax
	movslq	%ecx, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	history(%rip), %rdx
	movq	%rax, (%rcx,%rdx)
	addl	$1, -76(%rbp)
	movq	$95, -24(%rbp)
	jmp	.L173
.L131:
	addl	$1, -124(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L173
.L109:
	cmpl	$0, -104(%rbp)
	jne	.L184
	movq	$84, -24(%rbp)
	jmp	.L173
.L184:
	movq	$76, -24(%rbp)
	jmp	.L173
.L111:
	cmpl	$5, -120(%rbp)
	ja	.L186
	movl	-120(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L188(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L188(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L188:
	.long	.L186-.L188
	.long	.L192-.L188
	.long	.L191-.L188
	.long	.L190-.L188
	.long	.L189-.L188
	.long	.L187-.L188
	.text
.L187:
	movq	$11, -24(%rbp)
	jmp	.L193
.L189:
	movq	$56, -24(%rbp)
	jmp	.L193
.L190:
	movq	$74, -24(%rbp)
	jmp	.L193
.L191:
	movq	$15, -24(%rbp)
	jmp	.L193
.L192:
	movq	$89, -24(%rbp)
	jmp	.L193
.L186:
	movq	$17, -24(%rbp)
	nop
.L193:
	jmp	.L173
.L116:
	movl	history_count(%rip), %eax
	cmpl	%eax, -80(%rbp)
	jge	.L194
	movq	$59, -24(%rbp)
	jmp	.L173
.L194:
	movq	$38, -24(%rbp)
	jmp	.L173
.L160:
	addl	$1, -96(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L173
.L154:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC13(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -60(%rbp)
	movq	$37, -24(%rbp)
	jmp	.L173
.L99:
	movl	$2, -96(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L173
.L112:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	movl	%eax, -108(%rbp)
	movq	$53, -24(%rbp)
	jmp	.L173
.L118:
	movl	-96(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L196
	movq	$90, -24(%rbp)
	jmp	.L173
.L196:
	movq	$16, -24(%rbp)
	jmp	.L173
.L107:
	movl	-96(%rbp), %eax
	cmpl	$7, %eax
	ja	.L198
	movq	$68, -24(%rbp)
	jmp	.L173
.L198:
	movq	$14, -24(%rbp)
	jmp	.L173
.L93:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -48(%rbp)
	movl	-48(%rbp), %eax
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	exit@PLT
.L96:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC14(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -104(%rbp)
	movq	$81, -24(%rbp)
	jmp	.L173
.L164:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L200
	movq	$91, -24(%rbp)
	jmp	.L173
.L200:
	movq	$32, -24(%rbp)
	jmp	.L173
.L166:
	call	printDir2
	movl	$10, %edi
	call	putchar@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L163:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L202
	movq	$71, -24(%rbp)
	jmp	.L173
.L202:
	movq	$29, -24(%rbp)
	jmp	.L173
.L123:
	call	printDir2
	movl	$10, %edi
	call	putchar@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L134:
	cmpl	$0, -92(%rbp)
	jne	.L204
	movq	$94, -24(%rbp)
	jmp	.L173
.L204:
	movq	$0, -24(%rbp)
	jmp	.L173
.L157:
	movl	-124(%rbp), %eax
	cmpl	-128(%rbp), %eax
	jge	.L206
	movq	$65, -24(%rbp)
	jmp	.L173
.L206:
	movq	$44, -24(%rbp)
	jmp	.L173
.L148:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC14(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -52(%rbp)
	movq	$58, -24(%rbp)
	jmp	.L173
.L159:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L103:
	movl	-96(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -24(%rbp)
	jmp	.L173
.L119:
	movq	-32(%rbp), %rax
	leaq	.LC16(%rip), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	addq	$8, %rax
	leaq	.LC17(%rip), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	addq	$16, %rax
	leaq	.LC18(%rip), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	addq	$24, %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	addq	$32, %rax
	leaq	.LC20(%rip), %rdx
	movq	%rdx, (%rax)
	movl	$0, -124(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L173
.L130:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC21(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -72(%rbp)
	movq	$64, -24(%rbp)
	jmp	.L173
.L126:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L127:
	movl	-80(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -40(%rbp)
	movl	-80(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	history(%rip), %rax
	movq	(%rdx,%rax), %rdx
	movl	-40(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -80(%rbp)
	movq	$70, -24(%rbp)
	jmp	.L173
.L169:
	movl	$0, -80(%rbp)
	movq	$70, -24(%rbp)
	jmp	.L173
.L152:
	movl	$5, -128(%rbp)
	movl	$0, -120(%rbp)
	movq	$87, -24(%rbp)
	jmp	.L173
.L143:
	movl	$0, %eax
	jmp	.L208
.L125:
	movq	$27, -24(%rbp)
	jmp	.L173
.L106:
	movl	-128(%rbp), %eax
	cltq
	salq	$6, %rax
	addq	$63, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693944, %rax
	andq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %ecx
	movl	$0, %edx
	divq	%rcx
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L209:
	cmpq	%rdx, %rsp
	je	.L210
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L209
.L210:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L211
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L211:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -32(%rbp)
	movq	$67, -24(%rbp)
	jmp	.L173
.L128:
	cmpl	$0, -52(%rbp)
	jne	.L212
	movq	$50, -24(%rbp)
	jmp	.L173
.L212:
	movq	$100, -24(%rbp)
	jmp	.L173
.L108:
	call	helpcd
	movq	$38, -24(%rbp)
	jmp	.L173
.L146:
	movl	$0, history_count(%rip)
	movq	$38, -24(%rbp)
	jmp	.L173
.L114:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L214
	movq	$6, -24(%rbp)
	jmp	.L173
.L214:
	movq	$66, -24(%rbp)
	jmp	.L173
.L113:
	movl	-124(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -120(%rbp)
	movq	$44, -24(%rbp)
	jmp	.L173
.L137:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC24(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -56(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L173
.L115:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L155:
	movq	-136(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	movl	%eax, -112(%rbp)
	movq	$46, -24(%rbp)
	jmp	.L173
.L132:
	cmpl	$-1, -108(%rbp)
	jne	.L216
	movq	$4, -24(%rbp)
	jmp	.L173
.L216:
	movq	$38, -24(%rbp)
	jmp	.L173
.L121:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movq	-136(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -116(%rbp)
	movq	$8, -24(%rbp)
	jmp	.L173
.L138:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC12(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -92(%rbp)
	movq	$51, -24(%rbp)
	jmp	.L173
.L141:
	cmpl	$0, -120(%rbp)
	jne	.L218
	movq	$33, -24(%rbp)
	jmp	.L173
.L218:
	movq	$77, -24(%rbp)
	jmp	.L173
.L102:
	movl	$0, %edi
	call	exit@PLT
.L97:
	cmpl	$0, -84(%rbp)
	je	.L220
	movq	$18, -24(%rbp)
	jmp	.L173
.L220:
	movq	$38, -24(%rbp)
	jmp	.L173
.L95:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC14(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -64(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L173
.L147:
	movl	$1, %eax
	jmp	.L208
.L144:
	cmpl	$0, -60(%rbp)
	jne	.L222
	movq	$9, -24(%rbp)
	jmp	.L173
.L222:
	movq	$99, -24(%rbp)
	jmp	.L173
.L122:
	cmpl	$0, -72(%rbp)
	jne	.L224
	movq	$30, -24(%rbp)
	jmp	.L173
.L224:
	movq	$92, -24(%rbp)
	jmp	.L173
.L100:
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L98:
	movl	history_count(%rip), %eax
	cmpl	%eax, -76(%rbp)
	jge	.L226
	movq	$45, -24(%rbp)
	jmp	.L173
.L226:
	movq	$69, -24(%rbp)
	jmp	.L173
.L101:
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L165:
	movl	$2, -96(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L173
.L172:
	movl	$1, -96(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L173
.L139:
	cmpl	$-1, -112(%rbp)
	jne	.L228
	movq	$35, -24(%rbp)
	jmp	.L173
.L228:
	movq	$38, -24(%rbp)
	jmp	.L173
.L142:
	cmpl	$0, -100(%rbp)
	jne	.L230
	movq	$22, -24(%rbp)
	jmp	.L173
.L230:
	movq	$98, -24(%rbp)
	jmp	.L173
.L120:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC28(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -68(%rbp)
	movq	$31, -24(%rbp)
	jmp	.L173
.L168:
	cmpl	$0, -56(%rbp)
	jne	.L232
	movq	$63, -24(%rbp)
	jmp	.L173
.L232:
	movq	$24, -24(%rbp)
	jmp	.L173
.L105:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC24(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -100(%rbp)
	movq	$39, -24(%rbp)
	jmp	.L173
.L145:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$38, -24(%rbp)
	jmp	.L173
.L151:
	movq	-136(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC30(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -88(%rbp)
	movq	$20, -24(%rbp)
	jmp	.L173
.L171:
	cmpl	$0, -64(%rbp)
	jne	.L234
	movq	$52, -24(%rbp)
	jmp	.L173
.L234:
	movq	$60, -24(%rbp)
	jmp	.L173
.L156:
	cmpl	$0, -88(%rbp)
	jne	.L236
	movq	$10, -24(%rbp)
	jmp	.L173
.L236:
	movq	$47, -24(%rbp)
	jmp	.L173
.L240:
	nop
.L173:
	jmp	.L238
.L208:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L239
	call	__stack_chk_fail@PLT
.L239:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	ownCmdHandler, .-ownCmdHandler
	.section	.rodata
.LC31:
	.string	"lsh: allocation error\n"
.LC32:
	.string	" \t\r\n\007"
	.text
	.globl	split_line
	.type	split_line, @function
split_line:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$7, -24(%rbp)
.L268:
	cmpq	$19, -24(%rbp)
	ja	.L270
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L244(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L244(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L244:
	.long	.L257-.L244
	.long	.L270-.L244
	.long	.L256-.L244
	.long	.L255-.L244
	.long	.L270-.L244
	.long	.L270-.L244
	.long	.L254-.L244
	.long	.L253-.L244
	.long	.L252-.L244
	.long	.L251-.L244
	.long	.L270-.L244
	.long	.L250-.L244
	.long	.L249-.L244
	.long	.L270-.L244
	.long	.L248-.L244
	.long	.L270-.L244
	.long	.L247-.L244
	.long	.L246-.L244
	.long	.L245-.L244
	.long	.L243-.L244
	.text
.L245:
	movl	$64, -48(%rbp)
	movl	$0, -44(%rbp)
	movl	-48(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$8, -24(%rbp)
	jmp	.L258
.L248:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	%rax, (%rdx)
	addl	$1, -44(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L258
.L249:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$22, %edx
	movl	$1, %esi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L252:
	cmpq	$0, -40(%rbp)
	jne	.L259
	movq	$0, -24(%rbp)
	jmp	.L258
.L259:
	movq	$16, -24(%rbp)
	jmp	.L258
.L255:
	movq	-40(%rbp), %rax
	jmp	.L269
.L247:
	movq	-56(%rbp), %rax
	leaq	.LC32(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -32(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L258
.L250:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	$0, (%rax)
	movq	$3, -24(%rbp)
	jmp	.L258
.L251:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -32(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L258
.L243:
	cmpq	$0, -32(%rbp)
	je	.L262
	movq	$14, -24(%rbp)
	jmp	.L258
.L262:
	movq	$11, -24(%rbp)
	jmp	.L258
.L246:
	addl	$64, -48(%rbp)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L258
.L254:
	cmpq	$0, -40(%rbp)
	jne	.L264
	movq	$12, -24(%rbp)
	jmp	.L258
.L264:
	movq	$9, -24(%rbp)
	jmp	.L258
.L257:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$22, %edx
	movl	$1, %esi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L253:
	movq	$18, -24(%rbp)
	jmp	.L258
.L256:
	movl	-44(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jl	.L266
	movq	$17, -24(%rbp)
	jmp	.L258
.L266:
	movq	$9, -24(%rbp)
	jmp	.L258
.L270:
	nop
.L258:
	jmp	.L268
.L269:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	split_line, .-split_line
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
	subq	$1120, %rsp
	movl	%edi, -1092(%rbp)
	movq	%rsi, -1104(%rbp)
	movq	%rdx, -1112(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, history_count(%rip)
	nop
.L272:
	movq	$0, history(%rip)
	movq	$0, 8+history(%rip)
	movq	$0, 16+history(%rip)
	movq	$0, 24+history(%rip)
	movq	$0, 32+history(%rip)
	movq	$0, 40+history(%rip)
	movq	$0, 48+history(%rip)
	movq	$0, 56+history(%rip)
	movq	$0, 64+history(%rip)
	movq	$0, 72+history(%rip)
	movq	$0, 80+history(%rip)
	movq	$0, 88+history(%rip)
	movq	$0, 96+history(%rip)
	movq	$0, 104+history(%rip)
	movq	$0, 112+history(%rip)
	movq	$0, 120+history(%rip)
	movq	$0, 128+history(%rip)
	movq	$0, 136+history(%rip)
	movq	$0, 144+history(%rip)
	movq	$0, 152+history(%rip)
	movq	$0, 160+history(%rip)
	movq	$0, 168+history(%rip)
	movq	$0, 176+history(%rip)
	movq	$0, 184+history(%rip)
	movq	$0, 192+history(%rip)
	movq	$0, 200+history(%rip)
	movq	$0, 208+history(%rip)
	movq	$0, 216+history(%rip)
	movq	$0, 224+history(%rip)
	movq	$0, 232+history(%rip)
	movq	$0, 240+history(%rip)
	movq	$0, 248+history(%rip)
	movq	$0, 256+history(%rip)
	movq	$0, 264+history(%rip)
	movq	$0, 272+history(%rip)
	movq	$0, 280+history(%rip)
	movq	$0, 288+history(%rip)
	movq	$0, 296+history(%rip)
	movq	$0, 304+history(%rip)
	movq	$0, 312+history(%rip)
	movq	$0, 320+history(%rip)
	movq	$0, 328+history(%rip)
	movq	$0, 336+history(%rip)
	movq	$0, 344+history(%rip)
	movq	$0, 352+history(%rip)
	movq	$0, 360+history(%rip)
	movq	$0, 368+history(%rip)
	movq	$0, 376+history(%rip)
	movq	$0, 384+history(%rip)
	movq	$0, 392+history(%rip)
	movq	$0, 400+history(%rip)
	movq	$0, 408+history(%rip)
	movq	$0, 416+history(%rip)
	movq	$0, 424+history(%rip)
	movq	$0, 432+history(%rip)
	movq	$0, 440+history(%rip)
	movq	$0, 448+history(%rip)
	movq	$0, 456+history(%rip)
	movq	$0, 464+history(%rip)
	movq	$0, 472+history(%rip)
	movq	$0, 480+history(%rip)
	movq	$0, 488+history(%rip)
	movq	$0, 496+history(%rip)
	movq	$0, 504+history(%rip)
	movq	$0, 512+history(%rip)
	movq	$0, 520+history(%rip)
	movq	$0, 528+history(%rip)
	movq	$0, 536+history(%rip)
	movq	$0, 544+history(%rip)
	movq	$0, 552+history(%rip)
	movq	$0, 560+history(%rip)
	movq	$0, 568+history(%rip)
	movq	$0, 576+history(%rip)
	movq	$0, 584+history(%rip)
	movq	$0, 592+history(%rip)
	movq	$0, 600+history(%rip)
	movq	$0, 608+history(%rip)
	movq	$0, 616+history(%rip)
	movq	$0, 624+history(%rip)
	movq	$0, 632+history(%rip)
	movq	$0, 640+history(%rip)
	movq	$0, 648+history(%rip)
	movq	$0, 656+history(%rip)
	movq	$0, 664+history(%rip)
	movq	$0, 672+history(%rip)
	movq	$0, 680+history(%rip)
	movq	$0, 688+history(%rip)
	movq	$0, 696+history(%rip)
	movq	$0, 704+history(%rip)
	movq	$0, 712+history(%rip)
	movq	$0, 720+history(%rip)
	movq	$0, 728+history(%rip)
	movq	$0, 736+history(%rip)
	movq	$0, 744+history(%rip)
	movq	$0, 752+history(%rip)
	movq	$0, 760+history(%rip)
	movq	$0, 768+history(%rip)
	movq	$0, 776+history(%rip)
	movq	$0, 784+history(%rip)
	movq	$0, 792+history(%rip)
	movq	$0, 800+history(%rip)
	movq	$0, 808+history(%rip)
	movq	$0, 816+history(%rip)
	movq	$0, 824+history(%rip)
	movq	$0, 832+history(%rip)
	movq	$0, 840+history(%rip)
	movq	$0, 848+history(%rip)
	movq	$0, 856+history(%rip)
	movq	$0, 864+history(%rip)
	movq	$0, 872+history(%rip)
	movq	$0, 880+history(%rip)
	movq	$0, 888+history(%rip)
	movq	$0, 896+history(%rip)
	movq	$0, 904+history(%rip)
	movq	$0, 912+history(%rip)
	movq	$0, 920+history(%rip)
	movq	$0, 928+history(%rip)
	movq	$0, 936+history(%rip)
	movq	$0, 944+history(%rip)
	movq	$0, 952+history(%rip)
	movq	$0, 960+history(%rip)
	movq	$0, 968+history(%rip)
	movq	$0, 976+history(%rip)
	movq	$0, 984+history(%rip)
	movq	$0, 992+history(%rip)
	movq	$0, 1000+history(%rip)
	movq	$0, 1008+history(%rip)
	movq	$0, 1016+history(%rip)
	movq	$0, 1024+history(%rip)
	movq	$0, 1032+history(%rip)
	movq	$0, 1040+history(%rip)
	movq	$0, 1048+history(%rip)
	movq	$0, 1056+history(%rip)
	movq	$0, 1064+history(%rip)
	movq	$0, 1072+history(%rip)
	movq	$0, 1080+history(%rip)
	movq	$0, 1088+history(%rip)
	movq	$0, 1096+history(%rip)
	movq	$0, 1104+history(%rip)
	movq	$0, 1112+history(%rip)
	movq	$0, 1120+history(%rip)
	movq	$0, 1128+history(%rip)
	movq	$0, 1136+history(%rip)
	movq	$0, 1144+history(%rip)
	movq	$0, 1152+history(%rip)
	movq	$0, 1160+history(%rip)
	movq	$0, 1168+history(%rip)
	movq	$0, 1176+history(%rip)
	movq	$0, 1184+history(%rip)
	movq	$0, 1192+history(%rip)
	movq	$0, 1200+history(%rip)
	movq	$0, 1208+history(%rip)
	movq	$0, 1216+history(%rip)
	movq	$0, 1224+history(%rip)
	movq	$0, 1232+history(%rip)
	movq	$0, 1240+history(%rip)
	movq	$0, 1248+history(%rip)
	movq	$0, 1256+history(%rip)
	movq	$0, 1264+history(%rip)
	movq	$0, 1272+history(%rip)
	movq	$0, 1280+history(%rip)
	movq	$0, 1288+history(%rip)
	movq	$0, 1296+history(%rip)
	movq	$0, 1304+history(%rip)
	movq	$0, 1312+history(%rip)
	movq	$0, 1320+history(%rip)
	movq	$0, 1328+history(%rip)
	movq	$0, 1336+history(%rip)
	movq	$0, 1344+history(%rip)
	movq	$0, 1352+history(%rip)
	movq	$0, 1360+history(%rip)
	movq	$0, 1368+history(%rip)
	movq	$0, 1376+history(%rip)
	movq	$0, 1384+history(%rip)
	movq	$0, 1392+history(%rip)
	movq	$0, 1400+history(%rip)
	movq	$0, 1408+history(%rip)
	movq	$0, 1416+history(%rip)
	movq	$0, 1424+history(%rip)
	movq	$0, 1432+history(%rip)
	movq	$0, 1440+history(%rip)
	movq	$0, 1448+history(%rip)
	movq	$0, 1456+history(%rip)
	movq	$0, 1464+history(%rip)
	movq	$0, 1472+history(%rip)
	movq	$0, 1480+history(%rip)
	movq	$0, 1488+history(%rip)
	movq	$0, 1496+history(%rip)
	movq	$0, 1504+history(%rip)
	movq	$0, 1512+history(%rip)
	movq	$0, 1520+history(%rip)
	movq	$0, 1528+history(%rip)
	movq	$0, 1536+history(%rip)
	movq	$0, 1544+history(%rip)
	movq	$0, 1552+history(%rip)
	movq	$0, 1560+history(%rip)
	movq	$0, 1568+history(%rip)
	movq	$0, 1576+history(%rip)
	movq	$0, 1584+history(%rip)
	movq	$0, 1592+history(%rip)
	movq	$0, 1600+history(%rip)
	movq	$0, 1608+history(%rip)
	movq	$0, 1616+history(%rip)
	movq	$0, 1624+history(%rip)
	movq	$0, 1632+history(%rip)
	movq	$0, 1640+history(%rip)
	movq	$0, 1648+history(%rip)
	movq	$0, 1656+history(%rip)
	movq	$0, 1664+history(%rip)
	movq	$0, 1672+history(%rip)
	movq	$0, 1680+history(%rip)
	movq	$0, 1688+history(%rip)
	movq	$0, 1696+history(%rip)
	movq	$0, 1704+history(%rip)
	movq	$0, 1712+history(%rip)
	movq	$0, 1720+history(%rip)
	movq	$0, 1728+history(%rip)
	movq	$0, 1736+history(%rip)
	movq	$0, 1744+history(%rip)
	movq	$0, 1752+history(%rip)
	movq	$0, 1760+history(%rip)
	movq	$0, 1768+history(%rip)
	movq	$0, 1776+history(%rip)
	movq	$0, 1784+history(%rip)
	movq	$0, 1792+history(%rip)
	movq	$0, 1800+history(%rip)
	movq	$0, 1808+history(%rip)
	movq	$0, 1816+history(%rip)
	movq	$0, 1824+history(%rip)
	movq	$0, 1832+history(%rip)
	movq	$0, 1840+history(%rip)
	movq	$0, 1848+history(%rip)
	movq	$0, 1856+history(%rip)
	movq	$0, 1864+history(%rip)
	movq	$0, 1872+history(%rip)
	movq	$0, 1880+history(%rip)
	movq	$0, 1888+history(%rip)
	movq	$0, 1896+history(%rip)
	movq	$0, 1904+history(%rip)
	movq	$0, 1912+history(%rip)
	movq	$0, 1920+history(%rip)
	movq	$0, 1928+history(%rip)
	movq	$0, 1936+history(%rip)
	movq	$0, 1944+history(%rip)
	movq	$0, 1952+history(%rip)
	movq	$0, 1960+history(%rip)
	movq	$0, 1968+history(%rip)
	movq	$0, 1976+history(%rip)
	movq	$0, 1984+history(%rip)
	movq	$0, 1992+history(%rip)
	movq	$0, 2000+history(%rip)
	movq	$0, 2008+history(%rip)
	movq	$0, 2016+history(%rip)
	movq	$0, 2024+history(%rip)
	movq	$0, 2032+history(%rip)
	movq	$0, 2040+history(%rip)
	movq	$0, 2048+history(%rip)
	movq	$0, 2056+history(%rip)
	movq	$0, 2064+history(%rip)
	movq	$0, 2072+history(%rip)
	movq	$0, 2080+history(%rip)
	movq	$0, 2088+history(%rip)
	movq	$0, 2096+history(%rip)
	movq	$0, 2104+history(%rip)
	movq	$0, 2112+history(%rip)
	movq	$0, 2120+history(%rip)
	movq	$0, 2128+history(%rip)
	movq	$0, 2136+history(%rip)
	movq	$0, 2144+history(%rip)
	movq	$0, 2152+history(%rip)
	movq	$0, 2160+history(%rip)
	movq	$0, 2168+history(%rip)
	movq	$0, 2176+history(%rip)
	movq	$0, 2184+history(%rip)
	movq	$0, 2192+history(%rip)
	movq	$0, 2200+history(%rip)
	movq	$0, 2208+history(%rip)
	movq	$0, 2216+history(%rip)
	movq	$0, 2224+history(%rip)
	movq	$0, 2232+history(%rip)
	movq	$0, 2240+history(%rip)
	movq	$0, 2248+history(%rip)
	movq	$0, 2256+history(%rip)
	movq	$0, 2264+history(%rip)
	movq	$0, 2272+history(%rip)
	movq	$0, 2280+history(%rip)
	movq	$0, 2288+history(%rip)
	movq	$0, 2296+history(%rip)
	movq	$0, 2304+history(%rip)
	movq	$0, 2312+history(%rip)
	movq	$0, 2320+history(%rip)
	movq	$0, 2328+history(%rip)
	movq	$0, 2336+history(%rip)
	movq	$0, 2344+history(%rip)
	movq	$0, 2352+history(%rip)
	movq	$0, 2360+history(%rip)
	movq	$0, 2368+history(%rip)
	movq	$0, 2376+history(%rip)
	movq	$0, 2384+history(%rip)
	movq	$0, 2392+history(%rip)
	movq	$0, 2400+history(%rip)
	movq	$0, 2408+history(%rip)
	movq	$0, 2416+history(%rip)
	movq	$0, 2424+history(%rip)
	movq	$0, 2432+history(%rip)
	movq	$0, 2440+history(%rip)
	movq	$0, 2448+history(%rip)
	movq	$0, 2456+history(%rip)
	movq	$0, 2464+history(%rip)
	movq	$0, 2472+history(%rip)
	movq	$0, 2480+history(%rip)
	movq	$0, 2488+history(%rip)
	movq	$0, 2496+history(%rip)
	movq	$0, 2504+history(%rip)
	movq	$0, 2512+history(%rip)
	movq	$0, 2520+history(%rip)
	movq	$0, 2528+history(%rip)
	movq	$0, 2536+history(%rip)
	movq	$0, 2544+history(%rip)
	movq	$0, 2552+history(%rip)
	movq	$0, 2560+history(%rip)
	movq	$0, 2568+history(%rip)
	movq	$0, 2576+history(%rip)
	movq	$0, 2584+history(%rip)
	movq	$0, 2592+history(%rip)
	movq	$0, 2600+history(%rip)
	movq	$0, 2608+history(%rip)
	movq	$0, 2616+history(%rip)
	movq	$0, 2624+history(%rip)
	movq	$0, 2632+history(%rip)
	movq	$0, 2640+history(%rip)
	movq	$0, 2648+history(%rip)
	movq	$0, 2656+history(%rip)
	movq	$0, 2664+history(%rip)
	movq	$0, 2672+history(%rip)
	movq	$0, 2680+history(%rip)
	movq	$0, 2688+history(%rip)
	movq	$0, 2696+history(%rip)
	movq	$0, 2704+history(%rip)
	movq	$0, 2712+history(%rip)
	movq	$0, 2720+history(%rip)
	movq	$0, 2728+history(%rip)
	movq	$0, 2736+history(%rip)
	movq	$0, 2744+history(%rip)
	movq	$0, 2752+history(%rip)
	movq	$0, 2760+history(%rip)
	movq	$0, 2768+history(%rip)
	movq	$0, 2776+history(%rip)
	movq	$0, 2784+history(%rip)
	movq	$0, 2792+history(%rip)
	movq	$0, 2800+history(%rip)
	movq	$0, 2808+history(%rip)
	movq	$0, 2816+history(%rip)
	movq	$0, 2824+history(%rip)
	movq	$0, 2832+history(%rip)
	movq	$0, 2840+history(%rip)
	movq	$0, 2848+history(%rip)
	movq	$0, 2856+history(%rip)
	movq	$0, 2864+history(%rip)
	movq	$0, 2872+history(%rip)
	movq	$0, 2880+history(%rip)
	movq	$0, 2888+history(%rip)
	movq	$0, 2896+history(%rip)
	movq	$0, 2904+history(%rip)
	movq	$0, 2912+history(%rip)
	movq	$0, 2920+history(%rip)
	movq	$0, 2928+history(%rip)
	movq	$0, 2936+history(%rip)
	movq	$0, 2944+history(%rip)
	movq	$0, 2952+history(%rip)
	movq	$0, 2960+history(%rip)
	movq	$0, 2968+history(%rip)
	movq	$0, 2976+history(%rip)
	movq	$0, 2984+history(%rip)
	movq	$0, 2992+history(%rip)
	movq	$0, 3000+history(%rip)
	movq	$0, 3008+history(%rip)
	movq	$0, 3016+history(%rip)
	movq	$0, 3024+history(%rip)
	movq	$0, 3032+history(%rip)
	movq	$0, 3040+history(%rip)
	movq	$0, 3048+history(%rip)
	movq	$0, 3056+history(%rip)
	movq	$0, 3064+history(%rip)
	movq	$0, 3072+history(%rip)
	movq	$0, 3080+history(%rip)
	movq	$0, 3088+history(%rip)
	movq	$0, 3096+history(%rip)
	movq	$0, 3104+history(%rip)
	movq	$0, 3112+history(%rip)
	movq	$0, 3120+history(%rip)
	movq	$0, 3128+history(%rip)
	movq	$0, 3136+history(%rip)
	movq	$0, 3144+history(%rip)
	movq	$0, 3152+history(%rip)
	movq	$0, 3160+history(%rip)
	movq	$0, 3168+history(%rip)
	movq	$0, 3176+history(%rip)
	movq	$0, 3184+history(%rip)
	movq	$0, 3192+history(%rip)
	movq	$0, 3200+history(%rip)
	movq	$0, 3208+history(%rip)
	movq	$0, 3216+history(%rip)
	movq	$0, 3224+history(%rip)
	movq	$0, 3232+history(%rip)
	movq	$0, 3240+history(%rip)
	movq	$0, 3248+history(%rip)
	movq	$0, 3256+history(%rip)
	movq	$0, 3264+history(%rip)
	movq	$0, 3272+history(%rip)
	movq	$0, 3280+history(%rip)
	movq	$0, 3288+history(%rip)
	movq	$0, 3296+history(%rip)
	movq	$0, 3304+history(%rip)
	movq	$0, 3312+history(%rip)
	movq	$0, 3320+history(%rip)
	movq	$0, 3328+history(%rip)
	movq	$0, 3336+history(%rip)
	movq	$0, 3344+history(%rip)
	movq	$0, 3352+history(%rip)
	movq	$0, 3360+history(%rip)
	movq	$0, 3368+history(%rip)
	movq	$0, 3376+history(%rip)
	movq	$0, 3384+history(%rip)
	movq	$0, 3392+history(%rip)
	movq	$0, 3400+history(%rip)
	movq	$0, 3408+history(%rip)
	movq	$0, 3416+history(%rip)
	movq	$0, 3424+history(%rip)
	movq	$0, 3432+history(%rip)
	movq	$0, 3440+history(%rip)
	movq	$0, 3448+history(%rip)
	movq	$0, 3456+history(%rip)
	movq	$0, 3464+history(%rip)
	movq	$0, 3472+history(%rip)
	movq	$0, 3480+history(%rip)
	movq	$0, 3488+history(%rip)
	movq	$0, 3496+history(%rip)
	movq	$0, 3504+history(%rip)
	movq	$0, 3512+history(%rip)
	movq	$0, 3520+history(%rip)
	movq	$0, 3528+history(%rip)
	movq	$0, 3536+history(%rip)
	movq	$0, 3544+history(%rip)
	movq	$0, 3552+history(%rip)
	movq	$0, 3560+history(%rip)
	movq	$0, 3568+history(%rip)
	movq	$0, 3576+history(%rip)
	movq	$0, 3584+history(%rip)
	movq	$0, 3592+history(%rip)
	movq	$0, 3600+history(%rip)
	movq	$0, 3608+history(%rip)
	movq	$0, 3616+history(%rip)
	movq	$0, 3624+history(%rip)
	movq	$0, 3632+history(%rip)
	movq	$0, 3640+history(%rip)
	movq	$0, 3648+history(%rip)
	movq	$0, 3656+history(%rip)
	movq	$0, 3664+history(%rip)
	movq	$0, 3672+history(%rip)
	movq	$0, 3680+history(%rip)
	movq	$0, 3688+history(%rip)
	movq	$0, 3696+history(%rip)
	movq	$0, 3704+history(%rip)
	movq	$0, 3712+history(%rip)
	movq	$0, 3720+history(%rip)
	movq	$0, 3728+history(%rip)
	movq	$0, 3736+history(%rip)
	movq	$0, 3744+history(%rip)
	movq	$0, 3752+history(%rip)
	movq	$0, 3760+history(%rip)
	movq	$0, 3768+history(%rip)
	movq	$0, 3776+history(%rip)
	movq	$0, 3784+history(%rip)
	movq	$0, 3792+history(%rip)
	movq	$0, 3800+history(%rip)
	movq	$0, 3808+history(%rip)
	movq	$0, 3816+history(%rip)
	movq	$0, 3824+history(%rip)
	movq	$0, 3832+history(%rip)
	movq	$0, 3840+history(%rip)
	movq	$0, 3848+history(%rip)
	movq	$0, 3856+history(%rip)
	movq	$0, 3864+history(%rip)
	movq	$0, 3872+history(%rip)
	movq	$0, 3880+history(%rip)
	movq	$0, 3888+history(%rip)
	movq	$0, 3896+history(%rip)
	movq	$0, 3904+history(%rip)
	movq	$0, 3912+history(%rip)
	movq	$0, 3920+history(%rip)
	movq	$0, 3928+history(%rip)
	movq	$0, 3936+history(%rip)
	movq	$0, 3944+history(%rip)
	movq	$0, 3952+history(%rip)
	movq	$0, 3960+history(%rip)
	movq	$0, 3968+history(%rip)
	movq	$0, 3976+history(%rip)
	movq	$0, 3984+history(%rip)
	movq	$0, 3992+history(%rip)
	movq	$0, 4000+history(%rip)
	movq	$0, 4008+history(%rip)
	movq	$0, 4016+history(%rip)
	movq	$0, 4024+history(%rip)
	movq	$0, 4032+history(%rip)
	movq	$0, 4040+history(%rip)
	movq	$0, 4048+history(%rip)
	movq	$0, 4056+history(%rip)
	movq	$0, 4064+history(%rip)
	movq	$0, 4072+history(%rip)
	movq	$0, 4080+history(%rip)
	movq	$0, 4088+history(%rip)
	movq	$0, 4096+history(%rip)
	movq	$0, 4104+history(%rip)
	movq	$0, 4112+history(%rip)
	movq	$0, 4120+history(%rip)
	movq	$0, 4128+history(%rip)
	movq	$0, 4136+history(%rip)
	movq	$0, 4144+history(%rip)
	movq	$0, 4152+history(%rip)
	movq	$0, 4160+history(%rip)
	movq	$0, 4168+history(%rip)
	movq	$0, 4176+history(%rip)
	movq	$0, 4184+history(%rip)
	movq	$0, 4192+history(%rip)
	movq	$0, 4200+history(%rip)
	movq	$0, 4208+history(%rip)
	movq	$0, 4216+history(%rip)
	movq	$0, 4224+history(%rip)
	movq	$0, 4232+history(%rip)
	movq	$0, 4240+history(%rip)
	movq	$0, 4248+history(%rip)
	movq	$0, 4256+history(%rip)
	movq	$0, 4264+history(%rip)
	movq	$0, 4272+history(%rip)
	movq	$0, 4280+history(%rip)
	movq	$0, 4288+history(%rip)
	movq	$0, 4296+history(%rip)
	movq	$0, 4304+history(%rip)
	movq	$0, 4312+history(%rip)
	movq	$0, 4320+history(%rip)
	movq	$0, 4328+history(%rip)
	movq	$0, 4336+history(%rip)
	movq	$0, 4344+history(%rip)
	movq	$0, 4352+history(%rip)
	movq	$0, 4360+history(%rip)
	movq	$0, 4368+history(%rip)
	movq	$0, 4376+history(%rip)
	movq	$0, 4384+history(%rip)
	movq	$0, 4392+history(%rip)
	movq	$0, 4400+history(%rip)
	movq	$0, 4408+history(%rip)
	movq	$0, 4416+history(%rip)
	movq	$0, 4424+history(%rip)
	movq	$0, 4432+history(%rip)
	movq	$0, 4440+history(%rip)
	movq	$0, 4448+history(%rip)
	movq	$0, 4456+history(%rip)
	movq	$0, 4464+history(%rip)
	movq	$0, 4472+history(%rip)
	movq	$0, 4480+history(%rip)
	movq	$0, 4488+history(%rip)
	movq	$0, 4496+history(%rip)
	movq	$0, 4504+history(%rip)
	movq	$0, 4512+history(%rip)
	movq	$0, 4520+history(%rip)
	movq	$0, 4528+history(%rip)
	movq	$0, 4536+history(%rip)
	movq	$0, 4544+history(%rip)
	movq	$0, 4552+history(%rip)
	movq	$0, 4560+history(%rip)
	movq	$0, 4568+history(%rip)
	movq	$0, 4576+history(%rip)
	movq	$0, 4584+history(%rip)
	movq	$0, 4592+history(%rip)
	movq	$0, 4600+history(%rip)
	movq	$0, 4608+history(%rip)
	movq	$0, 4616+history(%rip)
	movq	$0, 4624+history(%rip)
	movq	$0, 4632+history(%rip)
	movq	$0, 4640+history(%rip)
	movq	$0, 4648+history(%rip)
	movq	$0, 4656+history(%rip)
	movq	$0, 4664+history(%rip)
	movq	$0, 4672+history(%rip)
	movq	$0, 4680+history(%rip)
	movq	$0, 4688+history(%rip)
	movq	$0, 4696+history(%rip)
	movq	$0, 4704+history(%rip)
	movq	$0, 4712+history(%rip)
	movq	$0, 4720+history(%rip)
	movq	$0, 4728+history(%rip)
	movq	$0, 4736+history(%rip)
	movq	$0, 4744+history(%rip)
	movq	$0, 4752+history(%rip)
	movq	$0, 4760+history(%rip)
	movq	$0, 4768+history(%rip)
	movq	$0, 4776+history(%rip)
	movq	$0, 4784+history(%rip)
	movq	$0, 4792+history(%rip)
	movq	$0, 4800+history(%rip)
	movq	$0, 4808+history(%rip)
	movq	$0, 4816+history(%rip)
	movq	$0, 4824+history(%rip)
	movq	$0, 4832+history(%rip)
	movq	$0, 4840+history(%rip)
	movq	$0, 4848+history(%rip)
	movq	$0, 4856+history(%rip)
	movq	$0, 4864+history(%rip)
	movq	$0, 4872+history(%rip)
	movq	$0, 4880+history(%rip)
	movq	$0, 4888+history(%rip)
	movq	$0, 4896+history(%rip)
	movq	$0, 4904+history(%rip)
	movq	$0, 4912+history(%rip)
	movq	$0, 4920+history(%rip)
	movq	$0, 4928+history(%rip)
	movq	$0, 4936+history(%rip)
	movq	$0, 4944+history(%rip)
	movq	$0, 4952+history(%rip)
	movq	$0, 4960+history(%rip)
	movq	$0, 4968+history(%rip)
	movq	$0, 4976+history(%rip)
	movq	$0, 4984+history(%rip)
	movq	$0, 4992+history(%rip)
	movq	$0, 5000+history(%rip)
	movq	$0, 5008+history(%rip)
	movq	$0, 5016+history(%rip)
	movq	$0, 5024+history(%rip)
	movq	$0, 5032+history(%rip)
	movq	$0, 5040+history(%rip)
	movq	$0, 5048+history(%rip)
	movq	$0, 5056+history(%rip)
	movq	$0, 5064+history(%rip)
	movq	$0, 5072+history(%rip)
	movq	$0, 5080+history(%rip)
	movq	$0, 5088+history(%rip)
	movq	$0, 5096+history(%rip)
	movq	$0, 5104+history(%rip)
	movq	$0, 5112+history(%rip)
	movq	$0, 5120+history(%rip)
	movq	$0, 5128+history(%rip)
	movq	$0, 5136+history(%rip)
	movq	$0, 5144+history(%rip)
	movq	$0, 5152+history(%rip)
	movq	$0, 5160+history(%rip)
	movq	$0, 5168+history(%rip)
	movq	$0, 5176+history(%rip)
	movq	$0, 5184+history(%rip)
	movq	$0, 5192+history(%rip)
	movq	$0, 5200+history(%rip)
	movq	$0, 5208+history(%rip)
	movq	$0, 5216+history(%rip)
	movq	$0, 5224+history(%rip)
	movq	$0, 5232+history(%rip)
	movq	$0, 5240+history(%rip)
	movq	$0, 5248+history(%rip)
	movq	$0, 5256+history(%rip)
	movq	$0, 5264+history(%rip)
	movq	$0, 5272+history(%rip)
	movq	$0, 5280+history(%rip)
	movq	$0, 5288+history(%rip)
	movq	$0, 5296+history(%rip)
	movq	$0, 5304+history(%rip)
	movq	$0, 5312+history(%rip)
	movq	$0, 5320+history(%rip)
	movq	$0, 5328+history(%rip)
	movq	$0, 5336+history(%rip)
	movq	$0, 5344+history(%rip)
	movq	$0, 5352+history(%rip)
	movq	$0, 5360+history(%rip)
	movq	$0, 5368+history(%rip)
	movq	$0, 5376+history(%rip)
	movq	$0, 5384+history(%rip)
	movq	$0, 5392+history(%rip)
	movq	$0, 5400+history(%rip)
	movq	$0, 5408+history(%rip)
	movq	$0, 5416+history(%rip)
	movq	$0, 5424+history(%rip)
	movq	$0, 5432+history(%rip)
	movq	$0, 5440+history(%rip)
	movq	$0, 5448+history(%rip)
	movq	$0, 5456+history(%rip)
	movq	$0, 5464+history(%rip)
	movq	$0, 5472+history(%rip)
	movq	$0, 5480+history(%rip)
	movq	$0, 5488+history(%rip)
	movq	$0, 5496+history(%rip)
	movq	$0, 5504+history(%rip)
	movq	$0, 5512+history(%rip)
	movq	$0, 5520+history(%rip)
	movq	$0, 5528+history(%rip)
	movq	$0, 5536+history(%rip)
	movq	$0, 5544+history(%rip)
	movq	$0, 5552+history(%rip)
	movq	$0, 5560+history(%rip)
	movq	$0, 5568+history(%rip)
	movq	$0, 5576+history(%rip)
	movq	$0, 5584+history(%rip)
	movq	$0, 5592+history(%rip)
	movq	$0, 5600+history(%rip)
	movq	$0, 5608+history(%rip)
	movq	$0, 5616+history(%rip)
	movq	$0, 5624+history(%rip)
	movq	$0, 5632+history(%rip)
	movq	$0, 5640+history(%rip)
	movq	$0, 5648+history(%rip)
	movq	$0, 5656+history(%rip)
	movq	$0, 5664+history(%rip)
	movq	$0, 5672+history(%rip)
	movq	$0, 5680+history(%rip)
	movq	$0, 5688+history(%rip)
	movq	$0, 5696+history(%rip)
	movq	$0, 5704+history(%rip)
	movq	$0, 5712+history(%rip)
	movq	$0, 5720+history(%rip)
	movq	$0, 5728+history(%rip)
	movq	$0, 5736+history(%rip)
	movq	$0, 5744+history(%rip)
	movq	$0, 5752+history(%rip)
	movq	$0, 5760+history(%rip)
	movq	$0, 5768+history(%rip)
	movq	$0, 5776+history(%rip)
	movq	$0, 5784+history(%rip)
	movq	$0, 5792+history(%rip)
	movq	$0, 5800+history(%rip)
	movq	$0, 5808+history(%rip)
	movq	$0, 5816+history(%rip)
	movq	$0, 5824+history(%rip)
	movq	$0, 5832+history(%rip)
	movq	$0, 5840+history(%rip)
	movq	$0, 5848+history(%rip)
	movq	$0, 5856+history(%rip)
	movq	$0, 5864+history(%rip)
	movq	$0, 5872+history(%rip)
	movq	$0, 5880+history(%rip)
	movq	$0, 5888+history(%rip)
	movq	$0, 5896+history(%rip)
	movq	$0, 5904+history(%rip)
	movq	$0, 5912+history(%rip)
	movq	$0, 5920+history(%rip)
	movq	$0, 5928+history(%rip)
	movq	$0, 5936+history(%rip)
	movq	$0, 5944+history(%rip)
	movq	$0, 5952+history(%rip)
	movq	$0, 5960+history(%rip)
	movq	$0, 5968+history(%rip)
	movq	$0, 5976+history(%rip)
	movq	$0, 5984+history(%rip)
	movq	$0, 5992+history(%rip)
	movq	$0, 6000+history(%rip)
	movq	$0, 6008+history(%rip)
	movq	$0, 6016+history(%rip)
	movq	$0, 6024+history(%rip)
	movq	$0, 6032+history(%rip)
	movq	$0, 6040+history(%rip)
	movq	$0, 6048+history(%rip)
	movq	$0, 6056+history(%rip)
	movq	$0, 6064+history(%rip)
	movq	$0, 6072+history(%rip)
	movq	$0, 6080+history(%rip)
	movq	$0, 6088+history(%rip)
	movq	$0, 6096+history(%rip)
	movq	$0, 6104+history(%rip)
	movq	$0, 6112+history(%rip)
	movq	$0, 6120+history(%rip)
	movq	$0, 6128+history(%rip)
	movq	$0, 6136+history(%rip)
	movq	$0, 6144+history(%rip)
	movq	$0, 6152+history(%rip)
	movq	$0, 6160+history(%rip)
	movq	$0, 6168+history(%rip)
	movq	$0, 6176+history(%rip)
	movq	$0, 6184+history(%rip)
	movq	$0, 6192+history(%rip)
	movq	$0, 6200+history(%rip)
	movq	$0, 6208+history(%rip)
	movq	$0, 6216+history(%rip)
	movq	$0, 6224+history(%rip)
	movq	$0, 6232+history(%rip)
	movq	$0, 6240+history(%rip)
	movq	$0, 6248+history(%rip)
	movq	$0, 6256+history(%rip)
	movq	$0, 6264+history(%rip)
	movq	$0, 6272+history(%rip)
	movq	$0, 6280+history(%rip)
	movq	$0, 6288+history(%rip)
	movq	$0, 6296+history(%rip)
	movq	$0, 6304+history(%rip)
	movq	$0, 6312+history(%rip)
	movq	$0, 6320+history(%rip)
	movq	$0, 6328+history(%rip)
	movq	$0, 6336+history(%rip)
	movq	$0, 6344+history(%rip)
	movq	$0, 6352+history(%rip)
	movq	$0, 6360+history(%rip)
	movq	$0, 6368+history(%rip)
	movq	$0, 6376+history(%rip)
	movq	$0, 6384+history(%rip)
	movq	$0, 6392+history(%rip)
	movq	$0, 6400+history(%rip)
	movq	$0, 6408+history(%rip)
	movq	$0, 6416+history(%rip)
	movq	$0, 6424+history(%rip)
	movq	$0, 6432+history(%rip)
	movq	$0, 6440+history(%rip)
	movq	$0, 6448+history(%rip)
	movq	$0, 6456+history(%rip)
	movq	$0, 6464+history(%rip)
	movq	$0, 6472+history(%rip)
	movq	$0, 6480+history(%rip)
	movq	$0, 6488+history(%rip)
	movq	$0, 6496+history(%rip)
	movq	$0, 6504+history(%rip)
	movq	$0, 6512+history(%rip)
	movq	$0, 6520+history(%rip)
	movq	$0, 6528+history(%rip)
	movq	$0, 6536+history(%rip)
	movq	$0, 6544+history(%rip)
	movq	$0, 6552+history(%rip)
	movq	$0, 6560+history(%rip)
	movq	$0, 6568+history(%rip)
	movq	$0, 6576+history(%rip)
	movq	$0, 6584+history(%rip)
	movq	$0, 6592+history(%rip)
	movq	$0, 6600+history(%rip)
	movq	$0, 6608+history(%rip)
	movq	$0, 6616+history(%rip)
	movq	$0, 6624+history(%rip)
	movq	$0, 6632+history(%rip)
	movq	$0, 6640+history(%rip)
	movq	$0, 6648+history(%rip)
	movq	$0, 6656+history(%rip)
	movq	$0, 6664+history(%rip)
	movq	$0, 6672+history(%rip)
	movq	$0, 6680+history(%rip)
	movq	$0, 6688+history(%rip)
	movq	$0, 6696+history(%rip)
	movq	$0, 6704+history(%rip)
	movq	$0, 6712+history(%rip)
	movq	$0, 6720+history(%rip)
	movq	$0, 6728+history(%rip)
	movq	$0, 6736+history(%rip)
	movq	$0, 6744+history(%rip)
	movq	$0, 6752+history(%rip)
	movq	$0, 6760+history(%rip)
	movq	$0, 6768+history(%rip)
	movq	$0, 6776+history(%rip)
	movq	$0, 6784+history(%rip)
	movq	$0, 6792+history(%rip)
	movq	$0, 6800+history(%rip)
	movq	$0, 6808+history(%rip)
	movq	$0, 6816+history(%rip)
	movq	$0, 6824+history(%rip)
	movq	$0, 6832+history(%rip)
	movq	$0, 6840+history(%rip)
	movq	$0, 6848+history(%rip)
	movq	$0, 6856+history(%rip)
	movq	$0, 6864+history(%rip)
	movq	$0, 6872+history(%rip)
	movq	$0, 6880+history(%rip)
	movq	$0, 6888+history(%rip)
	movq	$0, 6896+history(%rip)
	movq	$0, 6904+history(%rip)
	movq	$0, 6912+history(%rip)
	movq	$0, 6920+history(%rip)
	movq	$0, 6928+history(%rip)
	movq	$0, 6936+history(%rip)
	movq	$0, 6944+history(%rip)
	movq	$0, 6952+history(%rip)
	movq	$0, 6960+history(%rip)
	movq	$0, 6968+history(%rip)
	movq	$0, 6976+history(%rip)
	movq	$0, 6984+history(%rip)
	movq	$0, 6992+history(%rip)
	movq	$0, 7000+history(%rip)
	movq	$0, 7008+history(%rip)
	movq	$0, 7016+history(%rip)
	movq	$0, 7024+history(%rip)
	movq	$0, 7032+history(%rip)
	movq	$0, 7040+history(%rip)
	movq	$0, 7048+history(%rip)
	movq	$0, 7056+history(%rip)
	movq	$0, 7064+history(%rip)
	movq	$0, 7072+history(%rip)
	movq	$0, 7080+history(%rip)
	movq	$0, 7088+history(%rip)
	movq	$0, 7096+history(%rip)
	movq	$0, 7104+history(%rip)
	movq	$0, 7112+history(%rip)
	movq	$0, 7120+history(%rip)
	movq	$0, 7128+history(%rip)
	movq	$0, 7136+history(%rip)
	movq	$0, 7144+history(%rip)
	movq	$0, 7152+history(%rip)
	movq	$0, 7160+history(%rip)
	movq	$0, 7168+history(%rip)
	movq	$0, 7176+history(%rip)
	movq	$0, 7184+history(%rip)
	movq	$0, 7192+history(%rip)
	movq	$0, 7200+history(%rip)
	movq	$0, 7208+history(%rip)
	movq	$0, 7216+history(%rip)
	movq	$0, 7224+history(%rip)
	movq	$0, 7232+history(%rip)
	movq	$0, 7240+history(%rip)
	movq	$0, 7248+history(%rip)
	movq	$0, 7256+history(%rip)
	movq	$0, 7264+history(%rip)
	movq	$0, 7272+history(%rip)
	movq	$0, 7280+history(%rip)
	movq	$0, 7288+history(%rip)
	movq	$0, 7296+history(%rip)
	movq	$0, 7304+history(%rip)
	movq	$0, 7312+history(%rip)
	movq	$0, 7320+history(%rip)
	movq	$0, 7328+history(%rip)
	movq	$0, 7336+history(%rip)
	movq	$0, 7344+history(%rip)
	movq	$0, 7352+history(%rip)
	movq	$0, 7360+history(%rip)
	movq	$0, 7368+history(%rip)
	movq	$0, 7376+history(%rip)
	movq	$0, 7384+history(%rip)
	movq	$0, 7392+history(%rip)
	movq	$0, 7400+history(%rip)
	movq	$0, 7408+history(%rip)
	movq	$0, 7416+history(%rip)
	movq	$0, 7424+history(%rip)
	movq	$0, 7432+history(%rip)
	movq	$0, 7440+history(%rip)
	movq	$0, 7448+history(%rip)
	movq	$0, 7456+history(%rip)
	movq	$0, 7464+history(%rip)
	movq	$0, 7472+history(%rip)
	movq	$0, 7480+history(%rip)
	movq	$0, 7488+history(%rip)
	movq	$0, 7496+history(%rip)
	movq	$0, 7504+history(%rip)
	movq	$0, 7512+history(%rip)
	movq	$0, 7520+history(%rip)
	movq	$0, 7528+history(%rip)
	movq	$0, 7536+history(%rip)
	movq	$0, 7544+history(%rip)
	movq	$0, 7552+history(%rip)
	movq	$0, 7560+history(%rip)
	movq	$0, 7568+history(%rip)
	movq	$0, 7576+history(%rip)
	movq	$0, 7584+history(%rip)
	movq	$0, 7592+history(%rip)
	movq	$0, 7600+history(%rip)
	movq	$0, 7608+history(%rip)
	movq	$0, 7616+history(%rip)
	movq	$0, 7624+history(%rip)
	movq	$0, 7632+history(%rip)
	movq	$0, 7640+history(%rip)
	movq	$0, 7648+history(%rip)
	movq	$0, 7656+history(%rip)
	movq	$0, 7664+history(%rip)
	movq	$0, 7672+history(%rip)
	movq	$0, 7680+history(%rip)
	movq	$0, 7688+history(%rip)
	movq	$0, 7696+history(%rip)
	movq	$0, 7704+history(%rip)
	movq	$0, 7712+history(%rip)
	movq	$0, 7720+history(%rip)
	movq	$0, 7728+history(%rip)
	movq	$0, 7736+history(%rip)
	movq	$0, 7744+history(%rip)
	movq	$0, 7752+history(%rip)
	movq	$0, 7760+history(%rip)
	movq	$0, 7768+history(%rip)
	movq	$0, 7776+history(%rip)
	movq	$0, 7784+history(%rip)
	movq	$0, 7792+history(%rip)
	movq	$0, 7800+history(%rip)
	movq	$0, 7808+history(%rip)
	movq	$0, 7816+history(%rip)
	movq	$0, 7824+history(%rip)
	movq	$0, 7832+history(%rip)
	movq	$0, 7840+history(%rip)
	movq	$0, 7848+history(%rip)
	movq	$0, 7856+history(%rip)
	movq	$0, 7864+history(%rip)
	movq	$0, 7872+history(%rip)
	movq	$0, 7880+history(%rip)
	movq	$0, 7888+history(%rip)
	movq	$0, 7896+history(%rip)
	movq	$0, 7904+history(%rip)
	movq	$0, 7912+history(%rip)
	movq	$0, 7920+history(%rip)
	movq	$0, 7928+history(%rip)
	movq	$0, 7936+history(%rip)
	movq	$0, 7944+history(%rip)
	movq	$0, 7952+history(%rip)
	movq	$0, 7960+history(%rip)
	movq	$0, 7968+history(%rip)
	movq	$0, 7976+history(%rip)
	movq	$0, 7984+history(%rip)
	movq	$0, 7992+history(%rip)
	nop
.L273:
	movq	$0, _TIG_IZ_xoHD_envp(%rip)
	nop
.L274:
	movq	$0, _TIG_IZ_xoHD_argv(%rip)
	nop
.L275:
	movl	$0, _TIG_IZ_xoHD_argc(%rip)
	nop
	nop
.L276:
.L277:
#APP
# 1120 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-xoHD--0
# 0 "" 2
#NO_APP
	movl	-1092(%rbp), %eax
	movl	%eax, _TIG_IZ_xoHD_argc(%rip)
	movq	-1104(%rbp), %rax
	movq	%rax, _TIG_IZ_xoHD_argv(%rip)
	movq	-1112(%rbp), %rax
	movq	%rax, _TIG_IZ_xoHD_envp(%rip)
	nop
	movq	$5, -1064(%rbp)
.L288:
	cmpq	$7, -1064(%rbp)
	ja	.L290
	movq	-1064(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L280(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L280(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L280:
	.long	.L284-.L280
	.long	.L283-.L280
	.long	.L282-.L280
	.long	.L290-.L280
	.long	.L290-.L280
	.long	.L281-.L280
	.long	.L290-.L280
	.long	.L279-.L280
	.text
.L283:
	leaq	-1040(%rbp), %rdx
	movq	-1072(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	external_commands
	movq	$7, -1064(%rbp)
	jmp	.L285
.L281:
	movq	$2, -1064(%rbp)
	jmp	.L285
.L284:
	cmpl	$1, -1076(%rbp)
	jne	.L286
	movq	$1, -1064(%rbp)
	jmp	.L285
.L286:
	movq	$7, -1064(%rbp)
	jmp	.L285
.L279:
	call	printDir
	call	read_line
	movq	%rax, -1056(%rbp)
	movq	-1056(%rbp), %rax
	movq	%rax, %rdi
	call	add_history
	movq	-1056(%rbp), %rax
	movq	%rax, %rdi
	call	split_line
	movq	%rax, -1048(%rbp)
	movq	-1048(%rbp), %rax
	movq	%rax, -1072(%rbp)
	leaq	-1040(%rbp), %rdx
	movq	-1056(%rbp), %rcx
	movq	-1072(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	ownCmdHandler
	movl	%eax, -1076(%rbp)
	movq	$0, -1064(%rbp)
	jmp	.L285
.L282:
	leaq	-1040(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	getcwd@PLT
	call	init_shell
	movq	$7, -1064(%rbp)
	jmp	.L285
.L290:
	nop
.L285:
	jmp	.L288
	.cfi_endproc
.LFE10:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC33:
	.string	"\ncd: cd [-L| [dir]\nChange the shell working directory.\nChange the current directory to DIR. The default DIR is the value of the HOME shell variable."
	.align 8
.LC34:
	.string	"Options:\n-L\tforce symbolic links to be followed: resolve symbolic links in DIR after processing instances of `..'"
	.align 8
.LC35:
	.string	"`..' is processed by removing the immediately previous pathname component back to a slash or the beginning of DIR."
	.align 8
.LC36:
	.string	"Exit Status:\nReturns 0 if the directory is changed, and if $PWD is set successfully when -P is used; non-zero otherwise.\n"
	.text
	.globl	helpcd
	.type	helpcd, @function
helpcd:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L297:
	cmpq	$2, -8(%rbp)
	je	.L292
	cmpq	$2, -8(%rbp)
	ja	.L298
	cmpq	$0, -8(%rbp)
	je	.L299
	cmpq	$1, -8(%rbp)
	jne	.L298
	movq	$2, -8(%rbp)
	jmp	.L295
.L292:
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L295
.L298:
	nop
.L295:
	jmp	.L297
.L299:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	helpcd, .-helpcd
	.section	.rodata
	.align 8
.LC37:
	.string	"\npwd: pwd [-LP]\nPrint the name of the current working directory.\nOptions:\n-L print the value of $PWD if it names the current working directory"
	.align 8
.LC38:
	.string	"-P\tprint the physical directory, without any symbolic links"
	.align 8
.LC39:
	.string	"Exit Status:\nReturns 0 unless an invalid option is given or the current directory cannot be read.\n"
	.text
	.globl	helppwd
	.type	helppwd, @function
helppwd:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L306:
	cmpq	$2, -8(%rbp)
	je	.L301
	cmpq	$2, -8(%rbp)
	ja	.L307
	cmpq	$0, -8(%rbp)
	je	.L308
	cmpq	$1, -8(%rbp)
	jne	.L307
	movq	$2, -8(%rbp)
	jmp	.L304
.L301:
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L304
.L307:
	nop
.L304:
	jmp	.L306
.L308:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	helppwd, .-helppwd
	.section	.rodata
.LC40:
	.string	"Dir: %s$ "
	.text
	.globl	printDir
	.type	printDir, @function
printDir:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1056, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -1048(%rbp)
.L315:
	cmpq	$2, -1048(%rbp)
	je	.L310
	cmpq	$2, -1048(%rbp)
	ja	.L318
	cmpq	$0, -1048(%rbp)
	je	.L319
	cmpq	$1, -1048(%rbp)
	jne	.L318
	movq	$2, -1048(%rbp)
	jmp	.L313
.L310:
	leaq	-1040(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	getcwd@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -1048(%rbp)
	jmp	.L313
.L318:
	nop
.L313:
	jmp	.L315
.L319:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L317
	call	__stack_chk_fail@PLT
.L317:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	printDir, .-printDir
	.section	.rodata
.LC41:
	.string	"\033[H\033[J"
	.align 8
.LC42:
	.string	"\n\n\n\n******************************************"
.LC43:
	.string	"\n\n\n\t****MY SHELL****"
.LC44:
	.string	"\n\n\t-USE AT YOUR OWN RISK-"
	.text
	.globl	init_shell
	.type	init_shell, @function
init_shell:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L326:
	cmpq	$2, -8(%rbp)
	je	.L321
	cmpq	$2, -8(%rbp)
	ja	.L328
	cmpq	$0, -8(%rbp)
	je	.L323
	cmpq	$1, -8(%rbp)
	jne	.L328
	jmp	.L327
.L323:
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	movl	$1, %edi
	call	sleep@PLT
	movq	$1, -8(%rbp)
	jmp	.L325
.L321:
	movq	$0, -8(%rbp)
	jmp	.L325
.L328:
	nop
.L325:
	jmp	.L326
.L327:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	init_shell, .-init_shell
	.globl	add_history
	.type	add_history, @function
add_history:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$8, -8(%rbp)
.L345:
	cmpq	$8, -8(%rbp)
	ja	.L346
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L332(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L332(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L332:
	.long	.L338-.L332
	.long	.L337-.L332
	.long	.L347-.L332
	.long	.L346-.L332
	.long	.L335-.L332
	.long	.L334-.L332
	.long	.L333-.L332
	.long	.L346-.L332
	.long	.L331-.L332
	.text
.L335:
	movq	history(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L339
.L331:
	movl	history_count(%rip), %eax
	cmpl	$999, %eax
	jg	.L340
	movq	$6, -8(%rbp)
	jmp	.L339
.L340:
	movq	$4, -8(%rbp)
	jmp	.L339
.L337:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, 7992+history(%rip)
	movq	$2, -8(%rbp)
	jmp	.L339
.L333:
	movl	history_count(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	history_count(%rip), %eax
	addl	$1, %eax
	movl	%eax, history_count(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, %rcx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	history(%rip), %rax
	movq	%rcx, (%rdx,%rax)
	movq	$2, -8(%rbp)
	jmp	.L339
.L334:
	movl	-16(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	history(%rip), %rax
	movq	(%rdx,%rax), %rax
	movslq	%ecx, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	history(%rip), %rdx
	movq	%rax, (%rcx,%rdx)
	addl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L339
.L338:
	cmpl	$999, -16(%rbp)
	jg	.L342
	movq	$5, -8(%rbp)
	jmp	.L339
.L342:
	movq	$1, -8(%rbp)
	jmp	.L339
.L346:
	nop
.L339:
	jmp	.L345
.L347:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	add_history, .-add_history
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
